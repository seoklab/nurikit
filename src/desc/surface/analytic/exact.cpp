//
// Project NuriKit - Copyright 2026 SNU Compbio Lab.
// SPDX-License-Identifier: Apache-2.0
//

#if !defined(NURI_STRICT_FP) || defined(__FAST_MATH__)                         \
    || defined(__ASSOCIATIVE_MATH__) || defined(__RECIPROCAL_MATH__)
#error                                                                         \
    "exact.cpp must be compiled with NURI_STRICT_FP_FLAGS (src/CMakeLists.txt)"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

#include <absl/base/attributes.h>
#include <absl/log/absl_check.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <Eigen/Dense>
#include <geogram/numerics/multi_precision.h>

#include "nuri/eigen_config.h"
#include "nuri/desc/surface.h"

namespace nuri {
namespace internal {
  namespace {
    constexpr int kUnknown = 2;

    /**
     * Running error bound of a double computation: `|v − exact| ≤ e`. Each
     * rounding adds `u|v|` (`u = 2^-53`, taken as `2^-52` to cover `1/(1−u)`)
     * plus half an ulp of the smallest subnormal for results that underflow.
     * A sign is certified iff `|v| > 2e`; the factor 2 absorbs the rounding
     * of `e` itself for any number of operations below `2^50`.
     */
    struct Fx {
      double v, e;
    };

    constexpr double kUlp = 0x1p-52;
    constexpr double kTiny = 0x1p-1074;

    Fx operator+(Fx a, Fx b) {
      const double v = a.v + b.v;
      return { v, a.e + b.e + kUlp * std::fabs(v) + kTiny };
    }
    Fx operator-(Fx a, Fx b) {
      const double v = a.v - b.v;
      return { v, a.e + b.e + kUlp * std::fabs(v) + kTiny };
    }
    Fx operator*(Fx a, Fx b) {
      const double v = a.v * b.v;
      return { v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e
                      + kUlp * std::fabs(v) + kTiny };
    }
    Fx operator-(Fx a) {
      return { -a.v, a.e };
    }
    int sgn(Fx a) {
      if (a.e <= 0)
        return static_cast<int>(a.v > 0) - static_cast<int>(a.v < 0);
      if (a.v > 2 * a.e)
        return 1;
      if (a.v < -2 * a.e)
        return -1;
      return kUnknown;
    }

    /**
     * Exact dyadic number `m · 2^e`: every double is one, sums and products
     * stay exact, and no exponent range limits them (expansion arithmetic
     * underflows on degree-20 polynomials of rounded-zero coordinates).
     */
    class Xp {
    public:
      Xp(): m_(0), e_(0) { }

      explicit Xp(double d) {
        int exp = 0;
        const double fr = std::frexp(d, &exp);
        m_ = static_cast<std::int64_t>(std::ldexp(fr, 53));
        e_ = exp - 53;
        strip();
      }

      friend Xp operator+(const Xp &a, const Xp &b) {
        Xp r;
        r.e_ = std::min(a.e_, b.e_);
        r.m_ = (a.m_ << (a.e_ - r.e_)) + (b.m_ << (b.e_ - r.e_));
        r.strip();
        return r;
      }
      friend Xp operator-(const Xp &a, const Xp &b) {
        Xp r;
        r.e_ = std::min(a.e_, b.e_);
        r.m_ = (a.m_ << (a.e_ - r.e_)) - (b.m_ << (b.e_ - r.e_));
        r.strip();
        return r;
      }
      friend Xp operator*(const Xp &a, const Xp &b) {
        Xp r;
        r.m_ = a.m_ * b.m_;
        r.e_ = a.e_ + b.e_;
        return r;
      }
      friend Xp operator-(const Xp &a) {
        Xp r(a);
        r.m_ = -r.m_;
        return r;
      }
      friend int sgn(const Xp &a) { return a.m_.sign(); }

      double to_double() const {
        return std::ldexp(m_.convert_to<double>(), e_);
      }

    private:
      void strip() {
        if (m_.is_zero()) {
          e_ = 0;
          return;
        }
        const auto lsb = boost::multiprecision::lsb(abs(m_));
        m_ >>= lsb;
        e_ += static_cast<int>(lsb);
      }

      boost::multiprecision::cpp_int m_;
      int e_;
    };

    /**
     * Exact value and its derivative along one perturbed squared radius.
     */
    struct Dual {
      Xp v, d;
    };

    Dual operator+(const Dual &a, const Dual &b) {
      return { a.v + b.v, a.d + b.d };
    }
    Dual operator-(const Dual &a, const Dual &b) {
      return { a.v - b.v, a.d - b.d };
    }
    Dual operator*(const Dual &a, const Dual &b) {
      return { a.v * b.v, a.v * b.d + a.d * b.v };
    }
    Dual operator-(const Dual &a) {
      return { -a.v, -a.d };
    }

    template <class T>
    T lit(double d);
    template <>
    Fx lit<Fx>(double d) {
      return { d, 0.0 };
    }
    template <>
    Xp lit<Xp>(double d) {
      return Xp(d);
    }
    template <>
    Dual lit<Dual>(double d) {
      return { Xp(d), Xp(0.0) };
    }

    /**
     * Height of vertex `i`; `h_i = W + |c_i|² − ρ_i²`, so the perturbed
     * vertex has derivative −1 along its own squared radius.
     */
    template <class T>
    T height(double h, bool perturbed) {
      (void)perturbed;
      return lit<T>(h);
    }
    template <>
    Dual height<Dual>(double h, bool perturbed) {
      return { Xp(h), Xp(perturbed ? -1.0 : 0.0) };
    }

    template <class T>
    struct V3 {
      T x, y, z;
    };

    template <class T>
    V3<T> operator+(const V3<T> &a, const V3<T> &b) {
      return { a.x + b.x, a.y + b.y, a.z + b.z };
    }
    template <class T>
    V3<T> operator-(const V3<T> &a, const V3<T> &b) {
      return { a.x - b.x, a.y - b.y, a.z - b.z };
    }
    template <class T>
    V3<T> operator*(const T &s, const V3<T> &a) {
      return { s * a.x, s * a.y, s * a.z };
    }
    template <class T>
    T dot(const V3<T> &a, const V3<T> &b) {
      return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    template <class T>
    V3<T> cross(const V3<T> &a, const V3<T> &b) {
      return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
               a.x * b.y - a.y * b.x };
    }

    template <class T>
    struct Sphere {
      V3<T> c;
      T h;
    };

    struct Data {
      const Matrix3Xd *c;
      const ArrayXd *h;
      double w;
    };

    template <class T>
    struct Ctx {
      using Scalar = T;

      Data d;
      int pert;
    };

    template <class T>
    Sphere<T> sph(const Ctx<T> &ctx, int i) {
      const Matrix3Xd &c = *ctx.d.c;
      return {
        { lit<T>(c(0, i)), lit<T>(c(1, i)), lit<T>(c(2, i)) },
        height<T>((*ctx.d.h)[i], i == ctx.pert)
      };
    }

    template <class T>
    T wval(const Ctx<T> &ctx) {
      return lit<T>(ctx.d.w);
    }

    template <class T>
    T half(const T &a) {
      return lit<T>(0.5) * a;
    }

    /**
     * Circle `(a, b)` relative to `a`: `d = c_b − c_a`, `G = |d|²`, and
     * `π_b − π_a = 2(v − y·d)` for `y = x − c_a`. Its centre is
     * `c_a + (v / G) d`.
     */
    template <class T>
    struct Pair {
      V3<T> d;
      T g, v;
    };

    template <class T>
    Pair<T> pair(const Sphere<T> &sa, const Sphere<T> &sb) {
      V3<T> d = sb.c - sa.c;
      return { d, dot(d, d), half(sb.h - sa.h) - dot(sa.c, d) };
    }

    /**
     * Face `(a, b, c)` relative to `a`. The radical line is
     * `y = y_⊥ + s u`, `u = d_b × d_c`, `|u|² y_⊥ = λ d_b + μ d_c`, and the
     * cut points of circle `(a, b)` by `c` are `s = ±√D / |u|²` with
     * `D = |u|² (ρ_a² − |y_⊥|²)` written as a polynomial.
     */
    template <class T>
    struct Face {
      Sphere<T> sa;
      V3<T> db, dc, u;
      T gbb, gbc, gcc, vb, vc, lam, mu, u2, ra, disc;
    };

    template <class T>
    Face<T> face(const Ctx<T> &ctx, BallTriple f) {
      const Sphere<T> sa = sph(ctx, f.a), sb = sph(ctx, f.b),
                      sc = sph(ctx, f.c);
      Face<T> r;
      r.sa = sa;
      r.db = sb.c - sa.c;
      r.dc = sc.c - sa.c;
      r.u = cross(r.db, r.dc);
      r.gbb = dot(r.db, r.db);
      r.gbc = dot(r.db, r.dc);
      r.gcc = dot(r.dc, r.dc);
      r.u2 = dot(r.u, r.u);
      r.vb = half(sb.h - sa.h) - dot(sa.c, r.db);
      r.vc = half(sc.h - sa.h) - dot(sa.c, r.dc);
      r.lam = r.gcc * r.vb - r.gbc * r.vc;
      r.mu = r.gbb * r.vc - r.gbc * r.vb;
      r.ra = wval(ctx) + dot(sa.c, sa.c) - sa.h;
      r.disc = r.ra * r.u2
               - (r.gcc * r.vb * r.vb - lit<T>(2.0) * r.gbc * r.vb * r.vc
                  + r.gbb * r.vc * r.vc);
      return r;
    }

    template <class T>
    T root_sign(bool plus) {
      return lit<T>(plus ? 1.0 : -1.0);
    }

    /**
     * `A + B √D`, `D ≥ 0`; `B = D = 0` for a rational quantity.
     */
    template <class T>
    struct Root {
      T lin, rad, disc;
    };

    template <class T>
    int sign_root(const Root<T> &r) {
      const int sa = sgn(r.lin), sb = sgn(r.rad), sd = sgn(r.disc);
      if (sa == kUnknown || sb == kUnknown || sd == kUnknown)
        return kUnknown;

      ABSL_DCHECK_GE(sd, 0);
      if (sd == 0 || sb == 0)
        return sa;
      if (sa == 0)
        return sb;
      if (sa == sb)
        return sa;

      const int t = sgn(r.lin * r.lin - r.rad * r.rad * r.disc);
      return t == kUnknown ? kUnknown : sa * t;
    }

    /**
     * `2√D d/dε (A + B√D) = (2 B' D + B D') + 2 A' √D` for `D > 0`.
     */
    Root<Xp> first_order(const Root<Dual> &r) {
      const Xp two(2.0);
      return { two * r.rad.d * r.disc.v + r.rad.v * r.disc.d, two * r.lin.d,
               r.disc.v };
    }

    /**
     * Sign of `d/dε (A + B√D)` at an exact zero, with `D(ε) > 0`: for
     * `D > 0` it is `(2 A' √D + 2 B' D + B D') / (2√D)`; for `D = 0`
     * (a tangency perturbed to a cut) `√D(ε)` dominates every first-order
     * term unless `B = 0`.
     */
    int tie_sign(const Root<Dual> &r) {
      const int sd = sgn(r.disc.v);
      ABSL_DCHECK_GE(sd, 0);
      if (sd > 0)
        return sign_root(first_order(r));

      const int sb = sgn(r.rad.v);
      return sb != 0 ? sb : sgn(r.lin.d);
    }

    struct Participants {
      std::array<int, 8> v;
      int n;
    };

    /**
     * The perturbation over the participating vertices in index order: the
     * first non-zero coefficient decides.
     */
    template <class K, class Tie>
    Sgn perturbed(const Data &d, const K &kernel, Participants parts, Tie tie) {
      std::sort(parts.v.begin(), parts.v.begin() + parts.n);
      const int *end = std::unique(parts.v.begin(), parts.v.begin() + parts.n);
      for (const int *j = parts.v.begin(); j != end; ++j) {
        const int t = tie(kernel(Ctx<Dual> { d, *j }));
        if (t != 0)
          return static_cast<Sgn>(t);
      }
      ABSL_CHECK(false)
          << "every first-order perturbation coefficient vanishes";
      return Sgn::kZero;
    }

    /**
     * Filter, then exact, then the perturbation. `kernel(ctx)` returns
     * `Root<T>`.
     */
    template <bool kForceExact, class K>
    Sgn decide(const Data &d, const K &kernel, Participants parts) {
      if constexpr (!kForceExact) {
        const int s = sign_root(kernel(Ctx<Fx> { d, -1 }));
        if (s != kUnknown)
          return static_cast<Sgn>(s);
      }

      const int s = sign_root(kernel(Ctx<Xp> { d, -1 }));
      if (s != 0)
        return static_cast<Sgn>(s);
      return perturbed(d, kernel, parts, tie_sign);
    }

    /**
     * `(x0 + x1 √Di) + (y0 + y1 √Di) √Dj`; zero is reported, not perturbed.
     */
    template <class T>
    struct Root2 {
      T x0, x1, di, y0, y1, dj;
    };

    template <class T>
    int sign_root2(const Root2<T> &r) {
      const int sx = sign_root(Root<T> { r.x0, r.x1, r.di }),
                sy = sign_root(Root<T> { r.y0, r.y1, r.di }), sd = sgn(r.dj);
      if (sx == kUnknown || sy == kUnknown || sd == kUnknown)
        return kUnknown;

      ABSL_DCHECK_GE(sd, 0);
      if (sd == 0 || sy == 0)
        return sx;
      if (sx == 0)
        return sy;
      if (sx == sy)
        return sx;

      const T p = r.x0 * r.x0 + r.x1 * r.x1 * r.di
                  - (r.y0 * r.y0 + r.y1 * r.y1 * r.di) * r.dj,
              q = lit<T>(2.0) * (r.x0 * r.x1 - r.y0 * r.y1 * r.dj);
      const int t = sign_root(Root<T> { p, q, r.di });
      return t == kUnknown ? kUnknown : sx * t;
    }

    /**
     * First-order sign of `F = X + Y √Dj`, `X = x0 + x1 √Di`,
     * `Y = y0 + y1 √Di`, at an exact zero with both `D(ε) > 0`, along one
     * participant. For `Di, Dj > 0`, with `2√Di X' = A0 + A1 √Di` and
     * `2√Di Y' = B0 + B1 √Di`, `2√Di√Dj F' = √Dj (A0 + A1 √Di)
     * + Dj (B0 + B1 √Di) + Dj' (y0 √Di + y1 Di)`. One vanished discriminant
     * grows like `√ε`, so its coefficient, if non-zero, dominates every
     * first-order term whichever participant moves it.
     */
    int tie_sign2_one(const Root2<Dual> &r) {
      const int si = sgn(r.di.v), sj = sgn(r.dj.v);
      ABSL_DCHECK_GE(si, 0);
      ABSL_DCHECK_GE(sj, 0);
      ABSL_DCHECK(si > 0 || sj > 0);

      if (si > 0 && sj > 0) {
        const Root<Xp> a = first_order({ r.x0, r.x1, r.di }),
                       b = first_order({ r.y0, r.y1, r.di });
        return sign_root2(Root2<Xp> { r.dj.v * b.lin + r.dj.d * r.y1.v * r.di.v,
                                      r.dj.v * b.rad + r.dj.d * r.y0.v, r.di.v,
                                      a.lin, a.rad, r.dj.v });
      }
      if (si > 0) {
        const int sy = sign_root(Root<Xp> { r.y0.v, r.y1.v, r.di.v });
        return sy != 0 ? sy : tie_sign({ r.x0, r.x1, r.di });
      }
      const int sx = sign_root(Root<Xp> { r.x1.v, r.y1.v, r.dj.v });
      return sx != 0 ? sx : tie_sign({ r.x0, r.y0, r.dj });
    }

    /**
     * Both discriminants vanished; `rs[k]` carries the derivatives along the
     * `k`-th participant in grading order (`ρ_k²(ε) = ρ_k² + ε η^k`, `ε`
     * below every power of `η`), the values are shared. With
     * `Di(ε) = ε Li`, `Li = Σ_k Di_k' η^k`, and `Lj` likewise,
     * `F = √ε (x1 √Li + y0 √Lj) + ε Σ_k x0_k' η^k`: two roots tangent at an
     * exact zero have `P_i ∥ P_j`, hence `Q_i ∥ Q_j`, `y1 = 0` and
     * `x1, y0 ≠ 0`. The `√ε` term is decided by the discriminant that moves
     * first, or, when both first move at `j`, by `c = x1 √Di_j' + y0 √Dj_j'`
     * unless `c = 0`: then it is `c √η^j (√(1 + A) − √(1 + B))` for the
     * tails `A = Σ_{k>j} (Di_k' / Di_j') η^(k−j)` and `B` likewise, of the
     * sign of `c (A − B)`, so the first `k > j` with
     * `Di_k' Dj_j' ≠ Dj_k' Di_j'` decides; if none does, the `ε` term.
     */
    int tie_sign2_vanished(const Root2<Dual> *rs, const int n) {
      const Root2<Dual> &r0 = rs[0];
      const int sx = sgn(r0.x1.v), sy = sgn(r0.y0.v);
      ABSL_DCHECK(sx != 0 && sy != 0);
      ABSL_DCHECK_EQ(sgn(r0.y1.v), 0);

      for (int j = 0; j < n; ++j) {
        const Xp &ai = rs[j].di.d, &bi = rs[j].dj.d;
        const int sa = sgn(ai), sb = sgn(bi);
        if (sa == 0 && sb == 0)
          continue;
        ABSL_DCHECK_GE(sa, 0);
        ABSL_DCHECK_GE(sb, 0);
        if (sb == 0)
          return sx;
        if (sa == 0)
          return sy;

        const Xp zero(0.0);
        const int s =
            sign_root2(Root2<Xp> { zero, r0.x1.v, ai, r0.y0.v, zero, bi });
        if (s != 0)
          return s;
        for (int k = j + 1; k < n; ++k) {
          const int t = sgn(rs[k].di.d * bi - rs[k].dj.d * ai);
          if (t != 0)
            return sx * t;
        }
        break;
      }

      for (int k = 0; k < n; ++k) {
        const int s = sgn(rs[k].x0.d);
        if (s != 0)
          return s;
      }
      return 0;
    }

    template <class K>
    Sgn perturbed2(const Data &d, const K &kernel, Participants parts) {
      std::sort(parts.v.begin(), parts.v.begin() + parts.n);
      const int n = static_cast<int>(
          std::unique(parts.v.begin(), parts.v.begin() + parts.n)
          - parts.v.begin());

      std::array<Root2<Dual>, 8> rs;
      for (int k = 0; k < n; ++k)
        rs[k] = kernel(Ctx<Dual> { d, parts.v[k] });

      int t = 0;
      if (sgn(rs[0].di.v) == 0 && sgn(rs[0].dj.v) == 0) {
        t = tie_sign2_vanished(rs.data(), n);
      } else {
        for (int k = 0; k < n && t == 0; ++k)
          t = tie_sign2_one(rs[k]);
      }
      ABSL_CHECK(t != 0)
          << "every first-order perturbation coefficient vanishes";
      return static_cast<Sgn>(t);
    }

    template <bool kForceExact, class K>
    int sign2(const Data &d, const K &kernel) {
      if constexpr (!kForceExact) {
        const int s = sign_root2(kernel(Ctx<Fx> { d, -1 }));
        if (s != kUnknown)
          return s;
      }
      return sign_root2(kernel(Ctx<Xp> { d, -1 }));
    }

    template <bool kForceExact, class K>
    Sgn decide2(const Data &d, const K &kernel) {
      return static_cast<Sgn>(sign2<kForceExact>(d, kernel));
    }

    template <bool kForceExact, class K>
    Sgn decide2_perturbed(const Data &d, const K &kernel, Participants parts) {
      const int s = sign2<kForceExact>(d, kernel);
      if (s != 0)
        return static_cast<Sgn>(s);
      return perturbed2(d, kernel, parts);
    }

    double geogram_height(double x, double y, double z, double t) {
      const double tt = t * t, xx = x * x, yy = y * y, zz = z * z;
      const double s1 = tt + xx, s2 = s1 + yy;
      return s2 + zz;
    }

    V3<double> reference_ray(const Vector3d &d) {
      const int k = BallExact::reference_axis(d);
      V3<double> e { 0.0, 0.0, 0.0 };
      (k == 0 ? e.x : k == 1 ? e.y : e.z) = 1.0;
      return cross(V3<double> { d[0], d[1], d[2] }, e);
    }

    template <class T>
    V3<T> lift(const V3<double> &v) {
      return { lit<T>(v.x), lit<T>(v.y), lit<T>(v.z) };
    }

    /**
     * `G |u|² (x − cntr)` for root `x` of `f` on circle `(a, b)`:
     * `P + σ √D Q` with rational `P = G (|u|² (c_f − c_a) + λ d_b + μ d_c) −
     * |u|² v d` (`λ, μ, d_b, d_c` of the face relative to its first centre
     * `c_f`; `v, d, G` of the circle) and `Q = G u`, written in centre
     * differences so the filter's error bound is relative to the offset's
     * own length. `P ⊥ Q`.
     */
    template <class T>
    struct Offset {
      V3<T> p, q;
      T disc, u2, den;
    };

    template <class T>
    Offset<T> root_offset(const Ctx<T> &ctx, const Pair<T> &ab,
                          const Sphere<T> &sa, BallTriple f, bool plus) {
      const Face<T> fc = face(ctx, f);
      const V3<T> y = fc.u2 * (fc.sa.c - sa.c) + fc.lam * fc.db + fc.mu * fc.dc;
      return { ab.g * y - (fc.u2 * ab.v) * ab.d,
               (root_sign<T>(plus) * ab.g) * fc.u, fc.disc, fc.u2,
               ab.g * fc.u2 };
    }

    /**
     * `(P + √D Q) / den` in double when the running bound certifies the
     * vector to `kOffsetRelTol` of its length; `P ⊥ Q`, so the terms never
     * cancel and the bound is the sum of their errors.
     */
    bool certified_offset(const Offset<Fx> &o, Vector3d &out) {
      if (o.disc.e > 0 && o.disc.v <= 2 * o.disc.e)
        return false;
      const double s = std::sqrt(o.disc.v), es = s > 0 ? o.disc.e / s : 0;
      const Fx vs[3] = {
        o.p.x + Fx { s, es }
           * o.q.x, o.p.y + Fx { s, es }
           * o.q.y,
        o.p.z + Fx { s, es }
           * o.q.z
      };
      double err = 0, norm2 = 0;
      for (int i = 0; i < 3; ++i) {
        out[i] = vs[i].v;
        err += vs[i].e;
        norm2 += vs[i].v * vs[i].v;
      }
      if (err > kOffsetRelTol * std::sqrt(norm2))
        return false;
      out /= o.den.v;
      return true;
    }

    Vector3d exact_offset(const Offset<Xp> &o) {
      ABSL_DCHECK_GE(sgn(o.disc), 0);
      const double s = std::sqrt(o.disc.to_double());
      const Vector3d p(o.p.x.to_double(), o.p.y.to_double(), o.p.z.to_double()),
          q(o.q.x.to_double(), o.q.y.to_double(), o.q.z.to_double());
      return (p + s * q) / o.den.to_double();
    }
  }  // namespace

  namespace {
    /**
     * `T = ρ_a² + ρ_b² − d²`: overlap iff `T ≥ 0` or `4 ρ_a² ρ_b² − T² > 0`.
     * The second kernel is `4 G (ρ_a² − v_b² / G)`, the circle radius; an
     * exact zero is a tangency, a circle of zero length, and is reported so
     * the pair carries no cap.
     */
    template <bool kForceExact>
    Sgn overlap_impl(const Data &d, const int a, const int b) {
      auto stage = [&](auto ctx) {
        using T = typename decltype(ctx)::Scalar;
        const Sphere<T> sa = sph(ctx, a), sb = sph(ctx, b);
        const V3<T> dd = sb.c - sa.c;
        const T ra = wval(ctx) + dot(sa.c, sa.c) - sa.h,
                rb = wval(ctx) + dot(sb.c, sb.c) - sb.h;
        return std::pair<T, T> { ra + rb - dot(dd, dd), ra * rb };
      };
      auto kernel = [&](auto ctx) {
        using T = typename decltype(ctx)::Scalar;
        auto [t, rr] = stage(ctx);
        return Root<T> { lit<T>(4.0) * rr - t * t, lit<T>(0.0), lit<T>(0.0) };
      };

      if constexpr (!kForceExact) {
        const int s = sgn(stage(Ctx<Fx> { d, -1 }).first);
        if (s == 1 || s == 0)
          return Sgn::kPos;
        if (s == -1) {
          const int s2 = sign_root(kernel(Ctx<Fx> { d, -1 }));
          if (s2 != kUnknown && s2 != 0)
            return static_cast<Sgn>(s2);
        }
      }

      if (sgn(stage(Ctx<Xp> { d, -1 }).first) >= 0)
        return Sgn::kPos;
      return static_cast<Sgn>(sign_root(kernel(Ctx<Xp> { d, -1 })));
    }
  }  // namespace

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE BallExactImpl<kForceExact>
  BallExactImpl<kForceExact>::make(const Matrix4Xd &lifted, const double wmax) {
    BallExactImpl ex;
    const int n = static_cast<int>(lifted.cols());
    ex.c_ = lifted.topRows(3);
    ex.h_.resize(n);
    for (int i = 0; i < n; ++i)
      ex.h_[i] = geogram_height(lifted(0, i), lifted(1, i), lifted(2, i),
                                lifted(3, i));
    ex.w_ = wmax;
    return ex;
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE BallExactImpl<kForceExact>
  BallExactImpl<kForceExact>::make(const Matrix3Xd &centers,
                                   const ArrayXd &heights, const double wmax) {
    ABSL_DCHECK_EQ(centers.cols(), heights.size());
    BallExactImpl ex;
    ex.c_ = centers;
    ex.h_ = heights;
    ex.w_ = wmax;
    return ex;
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE double
  BallExactImpl<kForceExact>::rho2(const int i) const {
    const double cc =
        c_(0, i) * c_(0, i) + c_(1, i) * c_(1, i) + c_(2, i) * c_(2, i);
    return w_ + cc - h_[i];
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn
  BallExactImpl<kForceExact>::overlap(const int a, const int b) const {
    return overlap_impl<kForceExact>({ &c_, &h_, w_ }, a, b);
  }

  /**
   * `U = ρ_x² − ρ_y² − d²`: `B_y ⊂ B_x` iff `U ≥ 0` and `U² ≥ 4 ρ_y² d²`,
   * internal tangency included.
   */
  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE int
  BallExactImpl<kForceExact>::contained(const int a, const int b) const {
    const Data d { &c_, &h_, w_ };
    auto check = [&](auto ctx, const int x, const int y) {
      using T = typename decltype(ctx)::Scalar;
      const Sphere<T> sx = sph(ctx, x), sy = sph(ctx, y);
      const V3<T> dd = sy.c - sx.c;
      const T rx = wval(ctx) + dot(sx.c, sx.c) - sx.h,
              ry = wval(ctx) + dot(sy.c, sy.c) - sy.h, g = dot(dd, dd);
      const T u = rx - ry - g;
      const int su = sgn(u), sv = sgn(u * u - lit<T>(4.0) * ry * g);
      if (su == -1 || sv == -1)
        return 0;
      if (su == kUnknown || sv == kUnknown)
        return kUnknown;
      return 1;
    };
    auto inside = [&](const int x, const int y) {
      if constexpr (!kForceExact) {
        const int r = check(Ctx<Fx> { d, -1 }, x, y);
        if (r != kUnknown)
          return r == 1;
      }
      return check(Ctx<Xp> { d, -1 }, x, y) == 1;
    };
    const bool ab = inside(a, b), ba = inside(b, a);
    if (ab && ba)
      return 2;
    if (ab)
      return 1;
    if (ba)
      return 0;
    return -1;
  }

  /**
   * Circles `(a, b)` and `(a, c)` coincide iff `u = 0` (collinear centres)
   * and then `D = −gbb (k vb − vc)² = 0` for `d_c = k d_b` (equal centres).
   * The middle sphere is `a` iff `gbc < 0`, else the nearer of `b`, `c`.
   */
  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE int
  BallExactImpl<kForceExact>::shared_circle(const BallTriple f) const {
    const Data d { &c_, &h_, w_ };
    if constexpr (!kForceExact) {
      const Face<Fx> fc = face(Ctx<Fx> { d, -1 }, f);
      const int su = sgn(fc.u2), sd = sgn(fc.disc);
      if (su == 1 || sd == 1 || sd == -1)
        return -1;
    }

    const Face<Xp> fc = face(Ctx<Xp> { d, -1 }, f);
    if (sgn(fc.u2) != 0 || sgn(fc.disc) != 0)
      return -1;
    if (sgn(fc.gbc) < 0)
      return 0;
    return sgn(fc.gcc - fc.gbb) > 0 ? 1 : 2;
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn
  BallExactImpl<kForceExact>::cuts(const BallTriple f) const {
    auto kernel = [&](auto ctx) {
      using T = typename decltype(ctx)::Scalar;
      return Root<T> { face(ctx, f).disc, lit<T>(0.0), lit<T>(0.0) };
    };
    return decide<kForceExact>(
        {
            &c_, &h_, w_
    },
        kernel, { { f.a, f.b, f.c }, 3 });
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn BallExactImpl<kForceExact>::side(
      const int a, const int b, const int c) const {
    auto kernel = [&](auto ctx) {
      using T = typename decltype(ctx)::Scalar;
      return Root<T> { face(ctx, { a, b, c }).mu, lit<T>(0.0), lit<T>(0.0) };
    };
    return decide<kForceExact>(
        {
            &c_, &h_, w_
    },
        kernel, { { a, b, c }, 3 });
  }

  namespace {
    /**
     * Three signs from one face kernel relative to `s`: the discriminant of
     * the triple and `π_l` on circle `(s, j)`, `π_j` on circle `(s, l)`.
     */
    template <class T>
    std::array<Root<T>, 3> disc_roots(const Ctx<T> &ctx, int s, int j, int l) {
      const Face<T> fc = face(ctx, { s, j, l });
      const T zero = lit<T>(0.0);
      const T mu2 = fc.gcc * fc.vb - fc.gbc * fc.vc;
      return {
        Root<T> { fc.disc, zero, zero },
         Root<T> {   fc.mu, zero, zero },
        Root<T> {     mu2, zero, zero }
      };
    }

    /**
     * Intersect iff cut, else iff one circle is inside the other's ball; a
     * sign that stays unknown or ties is delegated to the exact `side`.
     */
    template <class T>
    int disc_decision(const std::array<Root<T>, 3> &r) {
      const int cut = sign_root(r[0]);
      if (cut == kUnknown || cut == 0)
        return kUnknown;
      if (cut > 0)
        return 1;

      const int s1 = sign_root(r[1]), s2 = sign_root(r[2]);
      if (s1 == -1 || s2 == -1)
        return 1;
      if (s1 == 1 && s2 == 1)
        return -1;
      return kUnknown;
    }
  }  // namespace

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE bool
  BallExactImpl<kForceExact>::discs_intersect(const int s, const int j,
                                              const int l) const {
    const Data d { &c_, &h_, w_ };
    if constexpr (!kForceExact) {
      const int r = disc_decision(disc_roots(Ctx<Fx> { d, -1 }, s, j, l));
      if (r != kUnknown)
        return r > 0;
    }
    const int r = disc_decision(disc_roots(Ctx<Xp> { d, -1 }, s, j, l));
    if (r != kUnknown)
      return r > 0;
    return side(s, j, l) == Sgn::kNeg || side(s, l, j) == Sgn::kNeg;
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn BallExactImpl<kForceExact>::accept(
      const BallTriple f, const bool plus, const int l) const {
    auto kernel = [&](auto ctx) {
      using T = typename decltype(ctx)::Scalar;
      const Face<T> fc = face(ctx, f);
      const Sphere<T> sl = sph(ctx, l);
      const V3<T> dl = sl.c - fc.sa.c;
      const T vl = half(sl.h - fc.sa.h) - dot(fc.sa.c, dl);
      const T lin =
          fc.u2 * vl - fc.lam * dot(fc.db, dl) - fc.mu * dot(fc.dc, dl);
      return Root<T> { lin, -root_sign<T>(plus) * dot(fc.u, dl), fc.disc };
    };
    return decide<kForceExact>(
        {
            &c_, &h_, w_
    },
        kernel, { { f.a, f.b, f.c, l }, 4 });
  }

  /**
   * Root as `cntr + offset`: the centre of circle `(a, b)` in double
   * and the radical-line offset certified by the running bound or taken
   * exact, so the direction from the centre is right to rounding on a
   * circle of any radius and for a third centre anywhere.
   */
  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Vector3d
  BallExactImpl<kForceExact>::root(const BallTriple f, const bool plus) const {
    const Vector3d ca = c_.col(f.a), db = c_.col(f.b) - ca;
    const double vb = (h_[f.b] - h_[f.a]) / 2 - ca.dot(db);
    const Vector3d cntr = ca + (vb / db.squaredNorm()) * db;
    return cntr + offset(f.a, f.b, f, plus);
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE std::pair<Vector3d, Vector3d>
  BallExactImpl<kForceExact>::roots(const BallTriple f) const {
    return { root(f, true), root(f, false) };
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Vector3d BallExactImpl<kForceExact>::offset(
      const int a, const int b, const BallTriple f, const bool plus) const {
    const Data d { &c_, &h_, w_ };
    auto kernel = [&](auto ctx) {
      using T = typename decltype(ctx)::Scalar;
      const Sphere<T> sa = sph(ctx, a), sb = sph(ctx, b);
      return root_offset(ctx, pair(sa, sb), sa, f, plus);
    };

    Vector3d out;
    if constexpr (!kForceExact) {
      if (certified_offset(kernel(Ctx<Fx> { d, -1 }), out))
        return out;
    }
    return exact_offset(kernel(Ctx<Xp> { d, -1 }));
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE int BallExactImpl<kForceExact>::half_plane(
      const int a, const int b, const BallTriple f, const bool plus) const {
    const V3<double> r = reference_ray(c_.col(b) - c_.col(a));
    const Data d { &c_, &h_, w_ };

    auto along = [&](const bool sine) {
      auto kernel = [&](auto ctx) {
        using T = typename decltype(ctx)::Scalar;
        const Sphere<T> sa = sph(ctx, a), sb = sph(ctx, b);
        const Pair<T> ab = pair(sa, sb);
        const Offset<T> o = root_offset(ctx, ab, sa, f, plus);
        const V3<T> rr = lift<T>(r), qv = sine ? cross(ab.d, rr) : rr;
        return Root<T> { dot(qv, o.p), dot(qv, o.q), o.disc };
      };
      if constexpr (!kForceExact) {
        const int s = sign_root(kernel(Ctx<Fx> { d, -1 }));
        if (s != kUnknown)
          return s;
      }
      return sign_root(kernel(Ctx<Xp> { d, -1 }));
    };

    const int s = along(true);
    if (s > 0)
      return 1;
    if (s < 0)
      return 3;
    return along(false) >= 0 ? 0 : 2;
  }

  namespace {
    /**
     * `(y_i × y_j) · d` for the offsets of roots `x_i`, `x_j` on circle
     * `(a, b)`, each `P + σ √D Q`.
     */
    template <class T>
    Root2<T> ccw_kernel(const Ctx<T> &ctx, int a, int b, BallTriple fi,
                        bool plus_i, BallTriple fj, bool plus_j) {
      const Sphere<T> sa = sph(ctx, a), sb = sph(ctx, b);
      const Pair<T> ab = pair(sa, sb);
      const Offset<T> oi = root_offset(ctx, ab, sa, fi, plus_i),
                      oj = root_offset(ctx, ab, sa, fj, plus_j);
      return {
        dot(cross(oi.p, oj.p), ab.d), dot(cross(oi.q, oj.p), ab.d), oi.disc,
        dot(cross(oi.p, oj.q), ab.d), dot(cross(oi.q, oj.q), ab.d), oj.disc
      };
    }
  }  // namespace

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn BallExactImpl<kForceExact>::ccw(
      const int a, const int b, const BallTriple fi, const bool plus_i,
      const BallTriple fj, const bool plus_j) const {
    auto kernel = [&](auto ctx) {
      return ccw_kernel(ctx, a, b, fi, plus_i, fj, plus_j);
    };
    return decide2<kForceExact>({ &c_, &h_, w_ }, kernel);
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn BallExactImpl<kForceExact>::ccw_perturbed(
      const int a, const int b, const BallTriple fi, const bool plus_i,
      const BallTriple fj, const bool plus_j) const {
    auto kernel = [&](auto ctx) {
      return ccw_kernel(ctx, a, b, fi, plus_i, fj, plus_j);
    };
    return decide2_perturbed<kForceExact>(
        {
            &c_, &h_, w_
    },
        kernel, { { a, b, fi.a, fi.b, fi.c, fj.a, fj.b, fj.c }, 8 });
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE Sgn BallExactImpl<kForceExact>::antipode(
      const int a, const int b, const BallTriple f, const bool plus,
      const int c) const {
    auto kernel = [&](auto ctx) {
      using T = typename decltype(ctx)::Scalar;
      const Sphere<T> sa = sph(ctx, a), sb = sph(ctx, b), sc = sph(ctx, c);
      const Pair<T> ab = pair(sa, sb), ac = pair(sa, sc);
      const Offset<T> o = root_offset(ctx, ab, sa, f, plus);
      const T lin =
          ab.g * o.u2 * ac.v - o.u2 * ab.v * dot(ab.d, ac.d) + dot(o.p, ac.d);
      return Root<T> { lin, dot(o.q, ac.d), o.disc };
    };
    return decide<kForceExact>(
        {
            &c_, &h_, w_
    },
        kernel, { { a, b, f.a, f.b, f.c, c }, 6 });
  }

  template <bool kForceExact>
  ABSL_ATTRIBUTE_NOINLINE bool BallExactImpl<kForceExact>::selftest() {
    double x, y;
    GEO::two_sum(1.0, 0x1p-60, x, y);
    if (x != 1.0 || y != 0x1p-60)
      return false;
    GEO::two_product(1.0 + 0x1p-30, 1.0 + 0x1p-30, x, y);
    if (y != 0x1p-60)
      return false;

    const Xp one(1.0), tiny(0x1p-60);
    if (sgn((one + tiny) - one) != 1)
      return false;

    const Fx f = (Fx { 1.0, 0.0 } + Fx { 0x1p-60, 0.0 }) - Fx { 1.0, 0.0 };
    return sgn(f) == kUnknown;
  }

  template class BallExactImpl<false>;
  template class BallExactImpl<true>;
}  // namespace internal
}  // namespace nuri
