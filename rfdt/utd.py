from __future__ import annotations

"""UTD Diffraction - Kouyoumjian-Pathak formulation using DrJit"""
import drjit as dr

from .constants import EPSILON_0, SMALL_EPS
from .rt_backend import Float, Complex2f

def _cot(x: Float) -> Float:
    y = dr.rcp(dr.tan(x))
    y = dr.select(dr.isnan(y), 0, y)
    y = dr.select(dr.isinf(y), 0, y)
    return y


def cot(x: Float, eps: float = SMALL_EPS) -> Float:
    sin_x, cos_x = dr.sincos(x)
    eps_f = Float(eps)
    denom = dr.select(dr.abs(sin_x) < eps_f, dr.sign(sin_x + eps_f) * eps_f, sin_x)
    y = cos_x / denom
    y = dr.select(dr.isnan(y), 0, y)
    y = dr.select(dr.isinf(y), 0, y)
    return y

def fresnel_integral(x: Float) -> Complex2f:
    """Fresnel integral using Boersma coefficients (loop unrolled for GPU efficiency)."""
    # Boersma coefficients for x < 4
    a0, a1, a2, a3 = +1.595769140, -0.000001702, -6.808568854, -0.000576361
    a4, a5, a6, a7 = +6.920691902, -0.016898657, -3.050485660, -0.075752419
    a8, a9, a10, a11 = +0.850663781, -0.025639041, -0.150230960, +0.034404779

    b0, b1, b2, b3 = -0.000000033, +4.255387524, -0.000092810, -7.780020400
    b4, b5, b6, b7 = -0.009520895, +5.075161298, -0.138341947, -1.363729124
    b8, b9, b10, b11 = -0.403349276, +0.702222016, -0.216195929, +0.019547031

    # Boersma coefficients for x >= 4
    c0, c1, c2, c3 = +0.000000000, -0.024933975, +0.000003936, +0.005770956
    c4, c5, c6, c7 = +0.000689892, -0.009497136, +0.011948809, -0.006748873
    c8, c9, c10, c11 = +0.000246420, +0.002102967, -0.001217930, +0.000233939

    d0, d1, d2, d3 = +0.199471140, +0.000000023, -0.009351341, +0.000023006
    d4, d5, d6, d7 = +0.004851466, +0.001903218, -0.017122914, +0.029064067
    d8, d9, d10, d11 = -0.027928955, +0.016497308, -0.005598515, +0.000838386

    x_pos = x > 0
    x = dr.abs(x)
    cond = x < 4
    arg = dr.select(cond, x * 0.25, 4 * dr.rcp(x))

    # Precompute powers of arg (unrolled)
    arg2 = arg * arg
    arg3 = arg2 * arg
    arg4 = arg2 * arg2
    arg5 = arg4 * arg
    arg6 = arg4 * arg2
    arg7 = arg6 * arg
    arg8 = arg4 * arg4
    arg9 = arg8 * arg
    arg10 = arg8 * arg2
    arg11 = arg8 * arg3

    # Compute polynomial using Horner-like grouping (unrolled, no Python loop)
    # r_part = sum(coef[n] * arg^n) for n=0..11
    r_coef = dr.select(cond, Float(a0), Float(c0)) + \
             dr.select(cond, Float(a1), Float(c1)) * arg + \
             dr.select(cond, Float(a2), Float(c2)) * arg2 + \
             dr.select(cond, Float(a3), Float(c3)) * arg3 + \
             dr.select(cond, Float(a4), Float(c4)) * arg4 + \
             dr.select(cond, Float(a5), Float(c5)) * arg5 + \
             dr.select(cond, Float(a6), Float(c6)) * arg6 + \
             dr.select(cond, Float(a7), Float(c7)) * arg7 + \
             dr.select(cond, Float(a8), Float(c8)) * arg8 + \
             dr.select(cond, Float(a9), Float(c9)) * arg9 + \
             dr.select(cond, Float(a10), Float(c10)) * arg10 + \
             dr.select(cond, Float(a11), Float(c11)) * arg11

    i_coef = dr.select(cond, Float(b0), Float(d0)) + \
             dr.select(cond, Float(b1), Float(d1)) * arg + \
             dr.select(cond, Float(b2), Float(d2)) * arg2 + \
             dr.select(cond, Float(b3), Float(d3)) * arg3 + \
             dr.select(cond, Float(b4), Float(d4)) * arg4 + \
             dr.select(cond, Float(b5), Float(d5)) * arg5 + \
             dr.select(cond, Float(b6), Float(d6)) * arg6 + \
             dr.select(cond, Float(b7), Float(d7)) * arg7 + \
             dr.select(cond, Float(b8), Float(d8)) * arg8 + \
             dr.select(cond, Float(b9), Float(d9)) * arg9 + \
             dr.select(cond, Float(b10), Float(d10)) * arg10 + \
             dr.select(cond, Float(b11), Float(d11)) * arg11

    arg_sqrt = dr.sqrt(arg)
    r_part = r_coef * arg_sqrt
    i_part = -i_coef * arg_sqrt

    sin_x, cos_x = dr.sincos(x)
    f_r = cos_x * r_part - sin_x * i_part
    f_i = cos_x * i_part + sin_x * r_part
    c_out = dr.select(cond, f_r, f_r + 0.5)
    s_out = dr.select(cond, f_i, f_i + 0.5)
    c_out = dr.select(x_pos, c_out, -c_out)
    s_out = dr.select(x_pos, s_out, -s_out)
    return Complex2f(c_out, s_out)

def f_utd(x: Float) -> Complex2f:
    """UTD transition function: F(x) = sqrt(πx/2) * e^(jx) * (1 + j - 2j*F_c*(x))"""
    f = Complex2f(1, 1)
    f -= Complex2f(0, 2) * dr.conj(fresnel_integral(x))
    f *= dr.sqrt(dr.pi * x / 2)
    f *= dr.exp(Complex2f(0, x))
    return f

def complex_sqrt(z: Complex2f) -> Complex2f:
    r, x, y = dr.abs(z), dr.real(z), dr.imag(z)
    return Complex2f(dr.sqrt((r + x) / 2), dr.sign(y) * dr.sqrt((r - x) / 2))

def complex_relative_permittivity(eta_r: Float, sigma: Float, omega: Float) -> Complex2f:
    return Complex2f(eta_r, -sigma * dr.rcp(omega * EPSILON_0))

def fresnel_reflection(cos_theta: Float, eta: Complex2f) -> tuple[Complex2f, Complex2f]:
    """Returns (r_te, r_tm) Fresnel reflection coefficients."""
    sin_theta_sqr = 1.0 - cos_theta * cos_theta
    a = complex_sqrt(eta - sin_theta_sqr)
    r_te = (cos_theta - a) * dr.rcp(cos_theta + a)
    r_tm = (eta * cos_theta - a) * dr.rcp(eta * cos_theta + a)
    return r_te, r_tm

def _compute_a_pm(beta: Float, n: Float) -> tuple[Float, Float]:
    two_n_pi = 2 * n * dr.pi
    n_plus = dr.round((beta + dr.pi) * dr.rcp(two_n_pi))
    n_minus = dr.round((beta - dr.pi) * dr.rcp(two_n_pi))
    a_plus = 2 * dr.cos((two_n_pi * n_plus - beta) / 2) ** 2
    a_minus = 2 * dr.cos((two_n_pi * n_minus - beta) / 2) ** 2
    return a_plus, a_minus

def diffraction_coefficient(phi: Float, phi_prime: Float, n: Float, k: Float, L: Float,
                            beta0: Float = None, R0: Complex2f = None, Rn: Complex2f = None) -> Complex2f:
    """UTD diffraction coefficient for PEC wedge. n = exterior_angle/π, L = s*s'/(s+s')*sin²(β₀)"""
    sin_beta0 = Float(1.0) if beta0 is None else dr.sin(beta0)
    factor = -dr.exp(Complex2f(0, -dr.pi / 4))
    factor *= dr.rcp(2 * n * dr.safe_sqrt(dr.two_pi * k) * sin_beta0)
    dif_phi, sum_phi = phi - phi_prime, phi + phi_prime
    a1, a2 = _compute_a_pm(dif_phi, n)
    a3, a4 = _compute_a_pm(sum_phi, n)
    two_n, kL = 2 * n, k * L
    d1 = cot((dr.pi + dif_phi) / two_n) * f_utd(kL * a1)
    d2 = cot((dr.pi - dif_phi) / two_n) * f_utd(kL * a2)
    d3 = cot((dr.pi + sum_phi) / two_n) * f_utd(kL * a3)
    d4 = cot((dr.pi - sum_phi) / two_n) * f_utd(kL * a4)
    R0 = Complex2f(-1, 0) if R0 is None else R0
    Rn = Complex2f(-1, 0) if Rn is None else Rn
    return factor * (d1 + d2 + R0 * d3 + Rn * d4)

def diffraction_coefficient_2d(phi: Float, phi_prime: Float, n: Float, k: Float,
                               s: Float, s_prime: Float, R0: Complex2f = None, Rn: Complex2f = None) -> Complex2f:
    """2D wrapper: computes L = s*s'/(s+s')"""
    L = s * s_prime * dr.rcp(s + s_prime)
    return diffraction_coefficient(phi, phi_prime, n, k, L, R0=R0, Rn=Rn)
