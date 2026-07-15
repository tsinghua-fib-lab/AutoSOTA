#pragma once

#include <iostream>
#include <cmath>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vector functions
// ||v1 - v2||_2
double compute_l2_distance_cuda(const double* d_vec1, const double* d_vec2, int size, cudaStream_t stream = cudaStreamPerThread);
// ||v||_2
double compute_l2_norm_cuda(const double* d_vec, int size, cudaStream_t stream = cudaStreamPerThread);
// log(x)
void compute_log_vector_cuda(const double* d_x, double* d_logx, int size, cudaStream_t stream = cudaStreamPerThread);
// x <- a * x
void compute_scaling_inplace_cuda(double a, double* d_x, int size, cudaStream_t stream = cudaStreamPerThread);
// z = a * x + y
void compute_axpy_cuda(const double* d_x, const double* d_y, double a, double* d_z, int size, cudaStream_t stream = cudaStreamPerThread);
// x'y
double compute_dot_prod_cuda(const double* d_x, const double* d_y, int size, cudaStream_t stream = cudaStreamPerThread);

// Overloaded versions using pinned host memory
double compute_l2_norm_cuda(const double* d_vec, double* h_pinned, int size, cudaStream_t stream = cudaStreamPerThread);
double compute_dot_prod_cuda(const double* d_x, const double* d_y, double* h_pinned, int size, cudaStream_t stream = cudaStreamPerThread);



// Use heuristics to set the total number of blocks
inline int heuristic_num_blocks()
{
    // Get the ID of the current active device
    int device_id = 0;
    cudaGetDevice(&device_id);

    // Get the number of streaming multiprocessors (SMs)
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
    // std::cout << "device_id = " << device_id << ", num_sms = " << num_sms << std::endl;
    // To avoid abnormal cases
    num_sms = std::max(num_sms, 10);
    // Target total number of blocks
    int target_num_blocks = 32 * num_sms;
    return target_num_blocks;
}

// Adjust number of blocks using heuristics
inline int heuristic_num_blocks(int init_num_blocks)
{
    // Target total number of blocks
    int target_num_blocks = heuristic_num_blocks();
    // Limit number of blocks to target_num_blocks to avoid
    // excessive kernel launch overhead
    return std::min(init_num_blocks, target_num_blocks);
}

// Adjust y-dimension with fixed x-dimension
inline int heuristic_num_blocks(int fixed_x, int init_y)
{
    // Target total number of blocks
    int target_num_blocks = heuristic_num_blocks();
    // Adjust y-dim according to x-dim
    int ymax = target_num_blocks / fixed_x;
    ymax = std::max(ymax, 1);
    return std::min(init_y, ymax);
}



// Functions used in line search

// Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
// that interpolates fa, ga, and fb, assuming the minimizer exists
// For case I: fb >= fa and ga * (b - a) < 0
inline double quadratic_minimizer(double a, double b, double fa, double ga, double fb)
{
    const double ba = b - a;
    const double w = 0.5 * ba * ga / (fa - fb + ba * ga);
    return a + w * ba;
}

// Minimizer of a quadratic function q(x) = c0 + c1 * x + c2 * x^2
// that interpolates fa, ga and gb, assuming the minimizer exists
// The result actually does not depend on fa
// For case II: ga * (b - a) < 0, ga * gb < 0
// For case III: ga * (b - a) < 0, ga * ga >= 0, |gb| <= |ga|
inline double quadratic_minimizer(double a, double b, double ga, double gb)
{
    const double w = ga / (ga - gb);
    return a + w * (b - a);
}

// Local minimizer of a cubic function q(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
// that interpolates fa, ga, fb and gb, assuming a != b
// Also sets a flag indicating whether the minimizer exists
inline double cubic_minimizer(
    double a, double b, double fa, double fb,
    double ga, double gb, bool& exists
)
{
    using std::abs;
    using std::sqrt;

    const double apb = a + b;
    const double ba = b - a;
    const double ba2 = ba * ba;
    const double fba = fb - fa;
    const double gba = gb - ga;
    // z3 = c3 * (b-a)^3, z2 = c2 * (b-a)^3, z1 = c1 * (b-a)^3
    const double z3 = (ga + gb) * ba - 2.0 * fba;
    const double z2 = 0.5 * (gba * ba2 - 3.0 * apb * z3);
    const double z1 = fba * ba2 - apb * z2 - (a * apb + b * b) * z3;

    // If c3 = z/(b-a)^3 == 0, reduce to quadratic problem
    constexpr double eps = std::numeric_limits<double>::epsilon();
    if (abs(z3) < eps * abs(z2) || abs(z3) < eps * abs(z1))
    {
        // Minimizer exists if c2 > 0
        exists = (z2 * ba > 0.0);
        // Return the end point if the minimizer does not exist
        return exists ? (-0.5 * z1 / z2) : b;
    }

    // Now we can assume z3 > 0
    // The minimizer is a solution to the equation c1 + 2*c2 * x + 3*c3 * x^2 = 0
    // roots = -(z2/z3) / 3 (+-) sqrt((z2/z3)^2 - 3 * (z1/z3)) / 3
    //
    // Let u = z2/(3z3) and v = z1/z2
    // The minimizer exists if v/u <= 1
    const double u = z2 / (3.0 * z3), v = z1 / z2;
    const double vu = v / u;
    exists = (vu <= 1.0);
    if (!exists)
        return b;

    // We need to find a numerically stable way to compute the roots, as z3 may still be small
    //
    // If |u| >= |v|, let w = 1 + sqrt(1-v/u), and then
    // r1 = -u * w, r2 = -v / w, r1 does not need to be the smaller one
    //
    // If |u| < |v|, we must have uv <= 0, and then
    // r = -u (+-) sqrt(delta), where
    // sqrt(delta) = sqrt(|u|) * sqrt(|v|) * sqrt(1-u/v)
    double r1 = 0.0, r2 = 0.0;
    if (abs(u) >= abs(v))
    {
        const double w = 1.0 + sqrt(1.0 - vu);
        r1 = -u * w;
        r2 = -v / w;
    }
    else
    {
        const double sqrtd = sqrt(abs(u)) * sqrt(abs(v)) * sqrt(1 - u / v);
        r1 = -u - sqrtd;
        r2 = -u + sqrtd;
    }
    return (z3 * ba > 0.0) ? ((std::max)(r1, r2)) : ((std::min)(r1, r2));
}

// Select the next step size according to the current step sizes,
// function values, and derivatives
inline double step_selection(
    double al, double au, double at,
    double fl, double fu, double ft,
    double gl, double gu, double gt
)
{
    using std::abs;

    if (al == au)
        return al;

    // If ft = Inf or gt = Inf, we return the middle point of al and at
    if (!std::isfinite(ft) || !std::isfinite(gt))
        return (al + at) / 2.0;

    // ac: cubic interpolation of fl, ft, gl, gt
    // aq: quadratic interpolation of fl, gl, ft
    bool ac_exists;
    const double ac = cubic_minimizer(al, at, fl, ft, gl, gt, ac_exists);
    const double aq = quadratic_minimizer(al, at, fl, gl, ft);
    // Case 1: ft > fl
    if (ft > fl)
    {
        // This should not happen if ft > fl, but just to be safe
        if (!ac_exists)
            return aq;
        // Then use the scheme described in the paper
        return (abs(ac - al) < abs(aq - al)) ? ac : ((aq + ac) / 2.0);
    }

    // as: quadratic interpolation of gl and gt
    const double as = quadratic_minimizer(al, at, gl, gt);
    // Case 2: ft <= fl, gt * gl < 0
    if (gt * gl < 0.0)
        return (abs(ac - at) >= abs(as - at)) ? ac : as;

    // Case 3: ft <= fl, gt * gl >= 0, |gt| < |gl|
    constexpr double deltal = 1.1, deltau = 0.66;
    if (abs(gt) < abs(gl))
    {
        // We choose either ac or as
        // The case for ac: 1. It exists, and
        //                  2. ac is farther than at from al, and
        //                  3. ac is closer to at than as
        // Cases for as: otherwise
        const bool choose_ac = ac_exists &&
            ((ac - at) * (at - al) > 0.0) &&
            (abs(ac - at) < abs(as - at));
        const double res = choose_ac ? ac : as;
        // Postprocessing the chosen step
        return (at > al) ?
            (std::min)(at + deltau * (au - at), res) :
            (std::max)(at + deltau * (au - at), res);
    }

    // Simple extrapolation if au, fu, or gu is infinity
    if ((!std::isfinite(au)) || (!std::isfinite(fu)) || (!std::isfinite(gu)))
        return at + deltal * (at - al);

    // ae: cubic interpolation of ft, fu, gt, gu
    bool ae_exists;
    const double ae = cubic_minimizer(at, au, ft, fu, gt, gu, ae_exists);
    // Case 4: ft <= fl, gt * gl >= 0, |gt| >= |gl|
    // The following is not used in the paper, but it seems to be a reasonable safeguard
    return (at > al) ?
        (std::min)(at + deltau * (au - at), ae) :
        (std::max)(at + deltau * (au - at), ae);
}
