/*
    Copyright (c) 2020 Andreas Finke <andreas.finke@unige.ch>

    All rights reserved. Use of this source code is governed by a modified BSD
    license that can be found in the LICENSE file.
*/


#pragma once

#include "mcmc.h"

#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/special.h>

constexpr int PACKET_SIZE = 16;
using FloatP = enoki::Packet<Float, PACKET_SIZE>;
using FloatX = enoki::DynamicArray<FloatP>;

#if PY == 1

struct ProbabilityDistributionSamples {

    ProbabilityDistributionSamples(const py::array_t<Float> samples, const py::array_t<Float> sigmas, bool dropLastForFullPackets) {

        /* round down to nearest multiple of PACKET_SIZE, throw away rest */

        int size;
        if (dropLastForFullPackets)
            size = int(samples.size()/PACKET_SIZE)*PACKET_SIZE;
        else
            size = samples.size();

        enoki::set_slices(y, size);
        enoki::set_slices(sig, size);

        auto y_access = samples.unchecked();
        auto sig_access = sigmas.unchecked();

        for (int i = 0; i < size; ++i) {
            enoki::slice(y, i) = y_access(i);
            enoki::slice(sig, i) = sig_access(i);
        }
    }
    FloatX y, sig;
};

class PiecewiseConstantPDF : public SubspaceState {

private:
    SumConstraint constraint;

public:

    Float lower, upper, binwidth;
    int nBins;

    PiecewiseConstantPDF(const ProbabilityDistributionSamples& data, Float lower, Float upper, int nBins) :
        SubspaceState({"pdf"}), constraint(nBins/(upper-lower)), lower(lower), upper(upper), nBins(nBins), data(data) {

        binwidth = (upper-lower)/nBins;

        setCoords({std::vector<Float>(nBins, Float(1)/(upper-lower))});

        constraint.link(&coords[0]);

        enoki::set_slices(pconv, enoki::slices(data.y));

    }

    void eval(const SharedParams& shared) override {

        FloatP loglikeP(0);

        for (size_t i = 0; i < enoki::packets(data.y); ++i) {
            packet(pconv, i) = prob_of_measurement(i);
            loglikeP += enoki::log(packet(pconv, i));
        };

        loglike = enoki::hsum(loglikeP);
    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = std::dynamic_pointer_cast<PiecewiseConstantPDF>(copy());
        //std::shared_ptr<PiecewiseConstantPDF> newstate = std::dynamic_pointer_cast<PiecewiseConstantPDF>(copy());

        /* if acceptance rate is high because data is not contraining enough individual bins, during adjustment, stepsizeCorrectionFac is growing without
         * bounds. Restrict to just the maximum (max over all pdfs) simultaneous minimal (min over all bins) value which is attained for the initial, flat, pdf. Together with the random float this guarantees acceptance of all steps in the flat-like case*/
        newstate->stepsizeCorrectionFac = std::min(stepsizeCorrectionFac, Float(1)/(upper-lower));

        int from, to;
        Float val, propRatio;
        newstate->constraint.step(rnd, newstate->stepsizeCorrectionFac, from, to, val, propRatio);

        //while (true) {
            //int from = std::min(int(rnd.nextFloat()*nBins), nBins-1);
            //int to   = std::min(int(rnd.nextFloat()*nBins), nBins-1);
            //Float val = newstate->stepsizeCorrectionFac*rnd.nextFloat();

            //if ( newstate->moveMass(from, to, val)) { //move allowed

                //Float newVol = newstate->accessibleStateVol(newstate->stepsizeCorrectionFac);
                //Float oldVol = newVol - newstate->accessibleStateVolCorrection(newstate->stepsizeCorrectionFac, from, to, val);

                /* the trans. probs q_ij from state i to state q are not symmetric, but obey
                 * q_ij  vol(reached from i) = q_ji  vol(reached from j)
                 * because sum_j q_ij = 1 is forcing q_ij to be inverse to the volume reached from i (i.e. that j is summed over)
                 *
                 * therefore, the factor q_ji/q_ij for the acceptance prob for going i->j is vol(from i)/vol(from j) = oldVol/newVol */

                //Float propRatio = oldVol/newVol;
                //float propRatio = 1;

                /* each pt contribute log( p_i) to log likelihood.
                 * p_i is the convolutio of the dist with a gaussian of width sig_i, approximated by
                 * p_i ~ binwidth * sum_over_bins_b_j { dist[j] * exp[- (b_j - y_i)^2 / 2 sig_i^2 ] / sqrt(2pi) / sig_i} where b_j is the position of the j-th bin
                 * assume we know p_i from before. if only two bins (from, to) changed by val, p_i changes as follows
                 * p_i -> p_i + Delta dist[to] * exp[- (b_to - y_i)^2/2sig_i^2]/sqrt(2pi)/sig_i + [replace "to" by "from"]
                 *     =  p_i + val { exp[-(b_to-y_i)^2/...] - exp[-(b_from-y_i)^2/...] } / sqrt(2pi) / sig_i
                 * more accurately, the j-th bin contributes to the convolution the following integral
                 *     dist[j] * \int_{b-halfwidth}^{b+halfwidth} dx exp(-(x-y_i)^2/2/sig_i^2)/sqrt(2pi)/sig_i =
                 *      dist[j]/2 ( erf((b-y + halfwidth)/sqrt(2)/sig) - erf((b-y - halfwidth)/sqrt(2)/sig) )
                 * so we get
                 * p_i -> p_i + Delta dist[to]/2 (erf(dto+ halfwidth) - erf(dto-halfwidth) ) + [replace "to" by "from"]
                 *
                 * note that Delta dist[to] = val, Delta dist[from] = -val

                */

                FloatP loglike_prop_packet(0);
                for (size_t i = 0; i < enoki::packets(data.y); ++i) {

                    FloatP sigi = enoki::packet(data.sig, i) + Float(0.000001);
                    FloatP invsig = 1/sigi;
                    FloatP f = invsig * Float(0.70710678118);
                    FloatP dto = binpos(to) - enoki::packet(data.y,i);
                    FloatP dfrom = binpos(from) - enoki::packet(data.y,i);
                    Float hw = binwidth*Float(0.5);
                    enoki::packet(newstate->pconv,i) = enoki::packet(pconv,i) + Float(0.5)*val*(enoki::erf((dto+hw)*f) - enoki::erf((dto-hw)*f) - enoki::erf((dfrom+hw)*f) + enoki::erf((dfrom-hw)*f));

                    loglike_prop_packet += enoki::log(enoki::packet(newstate->pconv,i));
                }

                newstate->loglike = enoki::hsum(loglike_prop_packet);

                return Proposal{newstate, propRatio};
            //}
        //}

    }

    std::shared_ptr<SubspaceState> copy() const override {
        auto foo = new PiecewiseConstantPDF(*this);
        auto& f = foo->getCoordsAt("pdf");
        foo->constraint.link(&f);
        //foo->constraint.link(&(foo->getCoords()[0]));
        return std::shared_ptr<SubspaceState>(foo);
    }

private:

    const ProbabilityDistributionSamples& data;

    FloatX pconv;

    Float binpos(int i) const {
        return lower + (i+Float(0.5))*binwidth;
    }


    FloatP prob_of_measurement(int packet) const {
        FloatP sigi = enoki::packet(data.sig, packet) + 0.000001;
        FloatP invsig = 1/sigi;
        FloatP f = invsig * 0.70710678118;

        FloatP prob(0);
        for (int j = 0; j < nBins; ++j) {
            Float hw = binwidth*Float(0.5);
            FloatP d = binpos(j) - enoki::packet(data.y, packet);
            prob += 0.5f*coords[0][j]*(enoki::erf((d+hw)*f) - enoki::erf((d-hw)*f));
        }
        return prob;
    }

};


class GaussianMixturePDF : public SubspaceState {

private:

    SumConstraint constraintAmplitudes;

public:

    Float lower, upper;
    size_t nModes;
    Float minSigma;
    Float maxSigma;

    bool usingShared = false;

    GaussianMixturePDF(const ProbabilityDistributionSamples& data, Float lower, Float upper, size_t nModes) :
        SubspaceState({"A", "mu", "sig", "nNonzeroModes"}, 1, false), constraintAmplitudes(1), lower(lower), upper(upper), nModes(nModes), data(&data) {

        minSigma = (upper-lower)/100;
        maxSigma = (upper-lower)*4;
        setCoords({std::vector<Float>(nModes, Float(1)/(nModes)),
                    std::vector<Float>(nModes, 0),
                     std::vector<Float>(nModes, (upper-lower)/std::min(size_t(4), nModes)), std::vector<Float>(1,nModes)});

        for (size_t i = 0; i < nModes; ++i)
            coords[1][i] = lower + (i+0.5)*(upper-lower)/(nModes);

        constraintAmplitudes.link(&coords[0]);

        usingShared = false;

    }

    std::string samples, errors;
    GaussianMixturePDF(const std::string& samples, const std::string& errors, Float lower, Float upper, size_t nModes) :
        SubspaceState({"A", "mu", "sig", "nNonzeroModes"}, 1, false), constraintAmplitudes(1), lower(lower), upper(upper), nModes(nModes), samples(samples), errors(errors) {

        requestedSharedNames.push_back(samples);
        requestedSharedNames.push_back(errors);

        minSigma = (upper-lower)/50;
        maxSigma = (upper-lower)*4;
        setCoords({std::vector<Float>(nModes, Float(1)/(nModes)),
                    std::vector<Float>(nModes, 0),
                     std::vector<Float>(nModes, (upper-lower)/std::min(size_t(4), nModes)), std::vector<Float>(1,nModes)});

        for (size_t i = 0; i < nModes; ++i)
            coords[1][i] = lower + (i+0.5)*(upper-lower)/(nModes);

        constraintAmplitudes.link(&coords[0]);

        usingShared = true;

    }

    void eval(const SharedParams& shared) override {

        if (usingShared == false) {
            FloatP loglikeP(0);

            for (size_t i = 0; i < enoki::packets(data->y); ++i) {

                FloatP p_i(0);
                for (size_t m = 0; m < nModes; ++m) {
                    FloatP var = enoki::packet(data->sig, i);
                    FloatP y   = enoki::packet(data->y, i);
                    FloatP arg = y - coords[1][m];
                    var *= var;
                    var += coords[2][m]*coords[2][m];
                    p_i += coords[0][m]/(enoki::sqrt(2*Float(3.14159265)*var))*enoki::exp(-arg*arg/(2*var));
                }
                loglikeP += enoki::log(p_i);
            };

            loglike = enoki::hsum(loglikeP);
        }
        else {
            loglike = 0;
            auto& y = shared.at(samples);
            auto& sig = shared.at(errors);

            for (size_t i = 0; i < y.size(); ++i) {
                Float p_i = 0;
                for (size_t m = 0; m < nModes; ++m) {
                    Float var = sig[i]*sig[i];
                    var += coords[2][m]*coords[2][m];
                    Float arg = y[i] - coords[1][m];
                    p_i += coords[0][m]/(std::sqrt(2*Float(3.14159265)*var))*std::exp(-arg*arg/(2*var));
                }
                loglike += std::log(p_i);
            }
        }

        coords[3][0] = 0;
        for (size_t i = 0; i < nModes; ++i)
            if (coords[0][i] > 0.005)
                coords[3][0] += 1;

    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = std::dynamic_pointer_cast<GaussianMixturePDF>(copy());

        Float propRatio = 1;

        /* sometimes mix various steps (and sometimes make them larger) to help get unstuck.
         * an easy way (not the fastest, but let's not worry about this) is to loop through multiple times.
         * large steps are not often accepted but should be tried a lot to help convergence. */
        int nSteps = rnd.nextFloat() * 10;

        for (int k = 0; k < nSteps; ++k) {
            Float stepkind = rnd.nextFloat();
            Float ampstepthresh = 0.5;
            Float otherthresh = 0.75;
            if (nModes == 1) {
                ampstepthresh = -0.1;
                otherthresh = 0.5;
            }

            if (stepkind < ampstepthresh) {
                int from, to;
                Float val;
                newstate->constraintAmplitudes.step(rnd, std::min(Float(0.1)*stepsizeCorrectionFac/nModes, Float(1.0)/nModes), from, to, val, propRatio);
            }
            else {

                //for (size_t i = 0; i < 1; ++i) {

                    size_t whichMode = std::min(size_t(rnd.nextFloat()*nModes), nModes-1);

                    if (stepkind < otherthresh) {

                        /* move at most by upper-lower (times 0.5) */
                        newstate->coords[1][whichMode] += (rnd.nextFloat()-0.5)*(upper-lower)*0.6*std::min(stepsizeCorrectionFac, Float(1));

                        /* reflect at boundary, based on the previous comment */
                        bound (newstate->coords[1][whichMode], lower, upper);
                        //if (newstate->coords[1][whichMode] < lower)
                            //newstate->coords[1][whichMode] = lower + (lower - newstate->coords[1][whichMode]);
                        //else if (newstate->coords[1][whichMode] > upper)
                            //newstate->coords[1][whichMode] = upper - (newstate->coords[1][whichMode] - upper);

                    }
                    else {

                        /*sometimes attempt large step: sigma might run away to too large values while amplitude is small, and then amplitude is stuck at 0
                         * if the data follows a narrowly peaked distribution.
                         * it is important to be able to have a shortcut to small sigma so amplitude can raise again in that case */
                        if (rnd.nextFloat() < 0.1f)
                            newstate->coords[2][whichMode] += (rnd.nextFloat()-0.5)*(maxSigma-minSigma)*std::min(stepsizeCorrectionFac, Float(1));
                        else
                            newstate->coords[2][whichMode] += (rnd.nextFloat()-0.5)*(maxSigma-minSigma)*0.05*std::min(stepsizeCorrectionFac, Float(1));

                        /*reflect at boundaries */
                        bound(newstate->coords[2][whichMode], minSigma, maxSigma);
                        //if (newstate->coords[2][whichMode] < minSigma)
                            //newstate->coords[2][whichMode] = minSigma + (minSigma - newstate->coords[2][whichMode]);
                        //else if (newstate->coords[2][whichMode] > maxSigma)
                            //newstate->coords[2][whichMode] = maxSigma - (newstate->coords[2][whichMode] - maxSigma);
                    }
                //}
            }
        }

        newstate->eval(shared);
        return Proposal{newstate, propRatio};

    }

    std::shared_ptr<SubspaceState> copy() const override {
        auto foo = new GaussianMixturePDF(*this);
        auto& f = foo->getCoordsAt("A");
        foo->constraintAmplitudes.link(&f);
        //foo->constraint.link(&(foo->getCoords()[0]));
        return std::shared_ptr<SubspaceState>(foo);
    }
private:

    const ProbabilityDistributionSamples* data;

};

template<typename T, typename T2>
T keelin_Q(const T& y, const std::vector<T2>& a) {
    T g = enoki::log(y/(1-y));
    T y5 = y - Float(0.5);
    return a[0] + a[1]*g + a[2]*y5*g + a[3]*y5 + a[4]*y5*y5 + a[5]*y5*y5*g + a[6]*y5*y5*y5 + a[7]*y5*y5*y5*g + a[8]*y5*y5*y5*y5 + a[9]*y5*y5*y5*y5*g;
}

template<typename T, typename T2>
T keelin_CDF(const T& x, const std::vector<T2>& a) {
    T high(1-1e-8);
    T low (1e-8);
    for (int i = 0; i < 20; ++i) {
        T mid = (high+low)*Float(0.5);
        T Qshifted = keelin_Q(mid, a) - x;
        enoki::masked(high, Qshifted>0) = mid;
        enoki::masked(low, Qshifted<0) = mid;
    }
    return (high+low)*Float(0.5);
}

template<typename T, typename T2>
T keelin_pdf(const T& x, const std::vector<T2>& a) {
    T y = keelin_CDF(x, a);
    T y1 = 1/(y*(1-y));
    T y5 = y - Float(0.5);
    T g = enoki::log(y/(1-y));
    T ret = 1/(a[1]*y1 + a[2]*(y5*y1 + g) + a[3] + a[4]*2*y5 + a[5]*(y5*y5*y1 + 2*y5*g) + a[6]*3*y5*y5 + a[7]*(y5*y5*y5*y1 + 3*y5*y5*g) + a[8]*4*y5*y5*y5 + a[9]*(y5*y5*y5*y5*y1 + 4*y5*y5*y5*g));
    return ret;
}
template<typename T, typename T2>
T keelin_pdf_of_y(const T& y, const std::vector<T2>& a) {
    T y1 = 1/(y*(1-y));
    T y5 = y - Float(0.5);
    T g = enoki::log(y/(1-y));
    T ret = 1/(a[1]*y1 + a[2]*(y5*y1 + g) + a[3] + a[4]*2*y5 + a[5]*(y5*y5*y1 + 2*y5*g) + a[6]*3*y5*y5 + a[7]*(y5*y5*y5*y1 + 3*y5*y5*g) + a[8]*4*y5*y5*y5 + a[9]*(y5*y5*y5*y5*y1 + 4*y5*y5*y5*g));
    return ret;
}

class KeelinPDF : public SubspaceState {

    Float std, var;
    Float mean;

    int nTerms = 5;

    size_t N = 0;

    void empiricalMeanStd() {
        FloatP meanP{0};
        for (size_t i = 0; i < enoki::packets(data->y); ++i) {
            meanP += enoki::packet(data->y, i);
        }
        mean = enoki::hsum(meanP)/N;

        FloatP varP{0};
        for (size_t i = 0; i < enoki::packets(data->y); ++i) {
            FloatP foo = enoki::packet(data->y, i) - mean;
            varP += foo*foo;
        }
        var = enoki::hsum(varP)/(N-1);

        std = std::sqrt(var);
    }

public:

    KeelinPDF(const ProbabilityDistributionSamples& data, int nTerms) :
        SubspaceState({"a", "adot"}, 0, false), data(&data), nTerms(nTerms) {

            empiricalMeanStd();

            N = enoki::slices(data.y);

            setCoords({std::vector<Float>{0, 1*std, 0, -3*std, 0, 0, 0, 0, 0, 0},
                    std::vector<Float>(10, Float(0))});

    }


    void eval(const SharedParams& shared) override {

        FloatP loglikeP(0);
        empiricalMeanStd();

        for (size_t i = 0; i < enoki::packets(data->y); ++i) {

            FloatP p_i(0);
            for (size_t j = nTerms; j < 10; ++j)
                coords[0][j] = 0;

            std::vector<FloatP> a{coords[0][0], coords[0][1], coords[0][2], coords[0][3], coords[0][4], coords[0][5], coords[0][6], coords[0][7], coords[0][8], coords[0][9]};
            for (size_t j = 0; j<nTerms; ++j) {
                // FloatP a = coords[0][j] + enoki::packet(data->sig, i)*cooords[1][j];
                // std::cout << j << " " << a[j];
                a[j] += enoki::packet(data->sig, i)*coords[1][j];
                // std::cout << "  " << a[j] << "\n";
            }
            FloatP x   = enoki::packet(data->y, i);
            p_i = keelin_pdf(x, a);
            p_i[p_i <= 0] = 1e-80; // unfeasible coefficients "a" manifest like so
            p_i[p_i > 1e3/std] = 1e-8; // no unfeasible but too weird
            loglikeP += enoki::log(p_i);
        }

        loglike = enoki::hsum(loglikeP);

        static constexpr auto pi = std::acos(-1);
        Float pi2 = pi*pi;
        Float pi4 = pi2*pi2;
        Float pi6 = pi2*pi4;

        Float a1 = coords[0][0];
        Float a2 = coords[0][1];
        Float a3 = coords[0][2];
        Float a4 = coords[0][3];
        Float a5 = coords[0][4];
        Float a6 = coords[0][5];
        Float a7 = coords[0][6];
        Float a8 = coords[0][7];

        Float m1 = a1 + a3/2. + a5/12. + a8/12.;
        Float m2 = a3*a3/12. + a2*a4 + a4*a4/12. + (a3*a5)/12. + a5*a5/180. + (2*a2*a6)/3. + (a4*a6)/6. + a6*a6/12. + (a2*a7)/6. + (a4*a7)/40. + (23*a6*a7)/720. + a7*a7/448. + (a3*a8)/12. + (13*a5*a8)/720. + a8*a8/80. + (a2*a2*pi2)/3. + (a3*a3*pi2)/36. + (a2*a6*pi2)/18. + (a6*a6*pi2)/240. + (a3*a8*pi2)/120. + (a8*a8*pi2)/1344.;
        Float m3 = (a2*a3*a4)/2. + (a3*a4*a4)/8. + a2*a2*a5 + (a3*a3*a5)/24. + \
(a2*a4*a5)/4. + (a4*a4*a5)/60. + (a3*a5*a5)/120. + a5*a5*a5/3780. + \
(a2*a3*a6)/2. + (a3*a4*a6)/4. + (a2*a5*a6)/3. + (13*a4*a5*a6)/240. + \
(a3*a6*a6)/8. + (3*a5*a6*a6)/80. + (a2*a3*a7)/4. + (7*a3*a4*a7)/120. \
+ (13*a2*a5*a7)/240. + (a4*a5*a7)/140. + (11*a3*a6*a7)/160. + \
(47*a5*a6*a7)/4032. + (29*a3*a7*a7)/4480. + (a5*a7*a7)/1344. + \
(3*a2*a2*a8)/4. + (a3*a3*a8)/24. + (a2*a4*a8)/4. + (13*a4*a4*a8)/480. \
+ (13*a3*a5*a8)/480. + (11*a5*a5*a8)/5040. + (a2*a6*a8)/3. + \
(3*a4*a6*a8)/40. + (23*a6*a6*a8)/480. + (3*a2*a7*a8)/40. + \
(3*a4*a7*a8)/224. + (1153*a6*a7*a8)/60480. + (59*a7*a7*a8)/38400. + \
(3*a3*a8*a8)/160. + (251*a5*a8*a8)/60480. + a8*a8*a8/448. + \
a2*a2*a3*pi2 + (a3*a3*a3*pi2)/24. + (a2*a3*a4*pi2)/6. + \
(a3*a3*a5*pi2)/180. + (5*a2*a3*a6*pi2)/12. + (a3*a4*a6*pi2)/40. + \
(a2*a5*a6*pi2)/90. + (a3*a6*a6*pi2)/24. + (a5*a6*a6*pi2)/840. + \
(a2*a3*a7*pi2)/40. + (a3*a6*a7*pi2)/224. + (a2*a2*a8*pi2)/6. + \
(41*a3*a3*a8*pi2)/1440. + (a2*a4*a8*pi2)/40. + (a3*a5*a8*pi2)/420. + \
(59*a2*a6*a8*pi2)/720. + (a4*a6*a8*pi2)/224. + \
(59*a6*a6*a8*pi2)/6720. + (a2*a7*a8*pi2)/224. + (a6*a7*a8*pi2)/1152. \
+ (89*a3*a8*a8*pi2)/13440. + (a5*a8*a8*pi2)/4032. + \
(59*a8*a8*a8*pi2)/115200.;

    Float m4 = a3*a3*a3*a3/80. + (a2*a3*a3*a4)/2. + 2*a2*a2*a4*a4 + \
(a3*a3*a4*a4)/8. + (a2*a4*a4*a4)/3. + a4*a4*a4*a4/80. + a2*a2*a3*a5 + \
(a3*a3*a3*a5)/24. + (5*a2*a3*a4*a5)/6. + (3*a3*a4*a4*a5)/40. + \
(a2*a2*a5*a5)/6. + (a3*a3*a5*a5)/45. + (a2*a4*a5*a5)/15. + \
(11*a4*a4*a5*a5)/2520. + (a3*a5*a5*a5)/420. + a5*a5*a5*a5/15120. + \
(2*a2*a3*a3*a6)/5. + 3*a2*a2*a4*a6 + (a3*a3*a4*a6)/4. + a2*a4*a4*a6 + \
(23*a4*a4*a4*a6)/360. + (5*a2*a3*a5*a6)/6. + (23*a3*a4*a5*a6)/120. + \
(17*a2*a5*a5*a6)/180. + (a4*a5*a5*a6)/70. + (6*a2*a2*a6*a6)/5. + \
(a3*a3*a6*a6)/8. + a2*a4*a6*a6 + (7*a4*a4*a6*a6)/60. + \
(7*a3*a5*a6*a6)/60. + (67*a5*a5*a6*a6)/6048. + (a2*a6*a6*a6)/3. + \
(11*a4*a6*a6*a6)/120. + (19*a6*a6*a6*a6)/720. + a2*a2*a2*a7 + \
(a2*a3*a3*a7)/4. + a2*a2*a4*a7 + (19*a3*a3*a4*a7)/240. + \
(23*a2*a4*a4*a7)/120. + (a4*a4*a4*a7)/112. + (23*a2*a3*a5*a7)/120. + \
(163*a3*a4*a5*a7)/5040. + (a2*a5*a5*a7)/70. + (a4*a5*a5*a7)/560. + \
a2*a2*a6*a7 + (43*a3*a3*a6*a7)/480. + (7*a2*a4*a6*a7)/15. + \
(11*a4*a4*a6*a7)/280. + (1391*a3*a5*a6*a7)/30240. + \
(239*a5*a5*a6*a7)/75600. + (11*a2*a6*a6*a7)/40. + \
(409*a4*a6*a6*a7)/7560. + (359*a6*a6*a6*a7)/15120. + \
(7*a2*a2*a7*a7)/60. + (1301*a3*a3*a7*a7)/120960. + \
(11*a2*a4*a7*a7)/280. + (a4*a4*a7*a7)/384. + (81*a3*a5*a7*a7)/22400. \
+ (17*a5*a5*a7*a7)/88704. + (409*a2*a6*a7*a7)/7560. + \
(563*a4*a6*a7*a7)/67200. + (141*a6*a6*a7*a7)/22400. + \
(563*a2*a7*a7*a7)/201600. + (a4*a7*a7*a7)/2816. + \
(1627*a6*a7*a7*a7)/2.66112e6 + a7*a7*a7*a7/53248. + \
(9*a2*a2*a3*a8)/10. + (a3*a3*a3*a8)/24. + (5*a2*a3*a4*a8)/6. + \
(23*a3*a4*a4*a8)/240. + (5*a2*a2*a5*a8)/12. + (79*a3*a3*a5*a8)/1440. \
+ (13*a2*a4*a5*a8)/60. + (179*a4*a4*a5*a8)/10080. + \
(37*a3*a5*a5*a8)/3024. + (43*a5*a5*a5*a8)/75600. + (5*a2*a3*a6*a8)/6. \
+ (7*a3*a4*a6*a8)/30. + (49*a2*a5*a6*a8)/180. + \
(155*a4*a5*a6*a8)/3024. + (11*a3*a6*a6*a8)/80. + \
(361*a5*a6*a6*a8)/10080. + (7*a2*a3*a7*a8)/30. + \
(187*a3*a4*a7*a8)/3780. + (155*a2*a5*a7*a8)/3024. + \
(263*a4*a5*a7*a8)/33600. + (1979*a3*a6*a7*a8)/30240. + \
(1591*a5*a6*a7*a8)/129600. + (839*a3*a7*a7*a8)/134400. + \
(7877*a5*a7*a7*a8)/8.8704e6 + (a2*a2*a8*a8)/4. + \
(47*a3*a3*a8*a8)/1440. + (3*a2*a4*a8*a8)/20. + (11*a4*a4*a8*a8)/756. \
+ (559*a3*a5*a8*a8)/30240. + (2971*a5*a5*a8*a8)/1.8144e6 + \
(8*a2*a6*a8*a8)/45. + (11*a4*a6*a8*a8)/280. + (29*a6*a6*a8*a8)/1120. \
+ (11*a2*a7*a8*a8)/280. + (17*a4*a7*a8*a8)/2400. + \
(37381*a6*a7*a8*a8)/3.6288e6 + (171*a7*a7*a8*a8)/197120. + \
(29*a3*a8*a8*a8)/3360. + (6827*a5*a8*a8*a8)/3.6288e6 + \
(43*a8*a8*a8*a8)/57600. + (3*a2*a2*a3*a3*pi2)/2. + \
(a3*a3*a3*a3*pi2)/24. + 2*a2*a2*a2*a4*pi2 + (2*a2*a3*a3*a4*pi2)/3. + \
(a2*a2*a4*a4*pi2)/6. + (a3*a3*a4*a4*pi2)/40. + (a2*a2*a3*a5*pi2)/2. + \
(a3*a3*a3*a5*pi2)/40. + (2*a2*a3*a4*a5*pi2)/45. + \
(a2*a2*a5*a5*pi2)/90. + (11*a3*a3*a5*a5*pi2)/7560. + \
(8*a2*a2*a2*a6*pi2)/3. + (13*a2*a3*a3*a6*pi2)/12. + a2*a2*a4*a6*pi2 + \
(17*a3*a3*a4*a6*pi2)/120. + (a2*a4*a4*a6*pi2)/20. + \
(7*a2*a3*a5*a6*pi2)/36. + (a3*a4*a5*a6*pi2)/105. + \
(11*a2*a5*a5*a6*pi2)/3780. + a2*a2*a6*a6*pi2 + \
(23*a3*a3*a6*a6*pi2)/160. + (23*a2*a4*a6*a6*pi2)/120. + \
(a4*a4*a6*a6*pi2)/224. + (211*a3*a5*a6*a6*pi2)/10080. + \
(a5*a5*a6*a6*pi2)/3360. + (7*a2*a6*a6*a6*pi2)/45. + \
(11*a4*a6*a6*a6*pi2)/840. + (409*a6*a6*a6*a6*pi2)/45360. + \
(a2*a2*a2*a7*pi2)/3. + (17*a2*a3*a3*a7*pi2)/120. + \
(a2*a2*a4*a7*pi2)/20. + (a3*a3*a4*a7*pi2)/112. + \
(a2*a3*a5*a7*pi2)/105. + (23*a2*a2*a6*a7*pi2)/120. + \
(17*a3*a3*a6*a7*pi2)/560. + (a2*a4*a6*a7*pi2)/56. + \
(a3*a5*a6*a7*pi2)/504. + (11*a2*a6*a6*a7*pi2)/280. + \
(a4*a6*a6*a7*pi2)/576. + (563*a6*a6*a6*a7*pi2)/201600. + \
(a2*a2*a7*a7*pi2)/224. + (a3*a3*a7*a7*pi2)/1152. + \
(a2*a6*a7*a7*pi2)/576. + (a6*a6*a7*a7*pi2)/5632. + \
(7*a2*a2*a3*a8*pi2)/6. + (7*a3*a3*a3*a8*pi2)/120. + \
(5*a2*a3*a4*a8*pi2)/18. + (a3*a4*a4*a8*pi2)/112. + \
(13*a2*a2*a5*a8*pi2)/120. + (101*a3*a3*a5*a8*pi2)/6048. + \
(a2*a4*a5*a8*pi2)/105. + (a3*a5*a5*a8*pi2)/1680. + \
(217*a2*a3*a6*a8*pi2)/360. + (103*a3*a4*a6*a8*pi2)/1680. + \
(649*a2*a5*a6*a8*pi2)/15120. + (a4*a5*a6*a8*pi2)/504. + \
(1129*a3*a6*a6*a8*pi2)/15120. + (949*a5*a6*a6*a8*pi2)/201600. + \
(103*a2*a3*a7*a8*pi2)/1680. + (a3*a4*a7*a8*pi2)/288. + \
(a2*a5*a7*a8*pi2)/504. + (341*a3*a6*a7*a8*pi2)/25200. + \
(a5*a6*a7*a8*pi2)/2376. + (a3*a7*a7*a8*pi2)/2816. + \
(59*a2*a2*a8*a8*pi2)/360. + (67*a3*a3*a8*a8*pi2)/2688. + \
(13*a2*a4*a8*a8*pi2)/420. + (a4*a4*a8*a8*pi2)/1152. + \
(769*a3*a5*a8*a8*pi2)/201600. + (17*a5*a5*a8*a8*pi2)/266112. + \
(11*a2*a6*a8*a8*pi2)/140. + (463*a4*a6*a8*a8*pi2)/67200. + \
(1913*a6*a6*a8*a8*pi2)/201600. + (463*a2*a7*a8*a8*pi2)/67200. + \
(a4*a7*a8*a8*pi2)/2816. + (4111*a6*a7*a8*a8*pi2)/2.66112e6 + \
(a7*a7*a8*a8*pi2)/26624. + (199*a3*a8*a8*a8*pi2)/44800. + \
(7877*a5*a8*a8*a8*pi2)/2.66112e7 + (57*a8*a8*a8*a8*pi2)/197120. + \
(7*a2*a2*a2*a2*pi4)/15. + (7*a2*a2*a3*a3*pi4)/30. + \
(7*a3*a3*a3*a3*pi4)/1200. + (7*a2*a2*a2*a6*pi4)/45. + \
(7*a2*a3*a3*a6*pi4)/100. + (7*a2*a2*a6*a6*pi4)/200. + \
(a3*a3*a6*a6*pi4)/160. + (a2*a6*a6*a6*pi4)/240. + \
(7*a6*a6*a6*a6*pi4)/34560. + (7*a2*a2*a3*a8*pi4)/100. + \
(a3*a3*a3*a8*pi4)/240. + (a2*a3*a6*a8*pi4)/40. + \
(7*a3*a6*a6*a8*pi4)/2880. + (a2*a2*a8*a8*pi4)/160. + \
(7*a3*a3*a8*a8*pi4)/5760. + (7*a2*a6*a8*a8*pi4)/2880. + \
(7*a6*a6*a8*a8*pi4)/28160. + (7*a3*a8*a8*a8*pi4)/42240. + \
(7*a8*a8*a8*a8*pi4)/798720.;
    Float m5 = (a2*a3*a3*a3*a4)/4. + (5*a2*a2*a3*a4*a4)/2. + (5*a3*a3*a3*a4*a4)/48. \
+ (5*a2*a3*a4*a4*a4)/6. + (7*a3*a4*a4*a4*a4)/144. + a2*a2*a3*a3*a5 + \
(a3*a3*a3*a3*a5)/48. + 5*a2*a2*a2*a4*a5 + (25*a2*a3*a3*a4*a5)/24. + \
(5*a2*a2*a4*a4*a5)/3. + (7*a3*a3*a4*a4*a5)/48. + \
(13*a2*a4*a4*a4*a5)/72. + (a4*a4*a4*a4*a5)/168. + \
(5*a2*a2*a3*a5*a5)/6. + (7*a3*a3*a3*a5*a5)/288. + \
(11*a2*a3*a4*a5*a5)/36. + (25*a3*a4*a4*a5*a5)/1008. + \
(a2*a2*a5*a5*a5)/18. + (a3*a3*a5*a5*a5)/189. + \
(11*a2*a4*a5*a5*a5)/756. + (a4*a4*a5*a5*a5)/1134. + \
(a3*a5*a5*a5*a5)/2835. + a5*a5*a5*a5*a5/149688. + (a2*a3*a3*a3*a6)/4. \
+ (9*a2*a2*a3*a4*a6)/2. + (5*a3*a3*a3*a4*a6)/24. + \
(5*a2*a3*a4*a4*a6)/2. + (11*a3*a4*a4*a4*a6)/48. + 4*a2*a2*a2*a5*a6 + \
(13*a2*a3*a3*a5*a6)/12. + (15*a2*a2*a4*a5*a6)/4. + \
(11*a3*a3*a4*a5*a6)/32. + (3*a2*a4*a4*a5*a6)/4. + \
(235*a4*a4*a4*a5*a6)/6048. + (59*a2*a3*a5*a5*a6)/144. + \
(227*a3*a4*a5*a5*a6)/3024. + (a2*a5*a5*a5*a6)/42. + \
(149*a4*a5*a5*a5*a6)/45360. + 2*a2*a2*a3*a6*a6 + \
(5*a3*a3*a3*a6*a6)/48. + (5*a2*a3*a4*a6*a6)/2. + \
(19*a3*a4*a4*a6*a6)/48. + 2*a2*a2*a5*a6*a6 + (19*a3*a3*a5*a6*a6)/96. \
+ (23*a2*a4*a5*a6*a6)/24. + (131*a4*a4*a5*a6*a6)/1512. + \
(643*a3*a5*a5*a6*a6)/12096. + (253*a5*a5*a5*a6*a6)/90720. + \
(5*a2*a3*a6*a6*a6)/6. + (43*a3*a4*a6*a6*a6)/144. + \
(7*a2*a5*a6*a6*a6)/18. + (487*a4*a5*a6*a6*a6)/6048. + \
(a3*a6*a6*a6*a6)/12. + (3*a5*a6*a6*a6*a6)/112. + \
(3*a2*a2*a2*a3*a7)/2. + (5*a2*a3*a3*a3*a7)/24. + \
(5*a2*a2*a3*a4*a7)/2. + (a3*a3*a3*a4*a7)/12. + \
(11*a2*a3*a4*a4*a7)/16. + (29*a3*a4*a4*a4*a7)/672. + \
(5*a2*a2*a2*a5*a7)/4. + (11*a2*a3*a3*a5*a7)/32. + \
(3*a2*a2*a4*a5*a7)/4. + (451*a3*a3*a4*a5*a7)/6048. + \
(235*a2*a4*a4*a5*a7)/2016. + (5*a4*a4*a4*a5*a7)/1008. + \
(227*a2*a3*a5*a5*a7)/3024. + (49*a3*a4*a5*a5*a7)/4320. + \
(149*a2*a5*a5*a5*a7)/45360. + (13*a4*a5*a5*a5*a7)/33264. + \
(5*a2*a2*a3*a6*a7)/2. + (53*a3*a3*a3*a6*a7)/576. + \
(19*a2*a3*a4*a6*a7)/12. + (521*a3*a4*a4*a6*a7)/3024. + \
(23*a2*a2*a5*a6*a7)/24. + (2357*a3*a3*a5*a6*a7)/24192. + \
(131*a2*a4*a5*a6*a7)/378. + (49*a4*a4*a5*a6*a7)/1920. + \
(3277*a3*a5*a5*a6*a7)/181440. + (4463*a5*a5*a5*a6*a7)/5.98752e6 + \
(43*a2*a3*a6*a6*a7)/48. + (167*a3*a4*a6*a6*a7)/756. + \
(487*a2*a5*a6*a6*a7)/2016. + (7331*a4*a5*a6*a6*a7)/181440. + \
(185*a3*a6*a6*a6*a7)/2016. + (4859*a5*a6*a6*a6*a7)/241920. + \
(19*a2*a2*a3*a7*a7)/48. + (653*a3*a3*a3*a7*a7)/48384. + \
(521*a2*a3*a4*a7*a7)/3024. + (97*a3*a4*a4*a7*a7)/6720. + \
(131*a2*a2*a5*a7*a7)/1512. + (6751*a3*a3*a5*a7*a7)/725760. + \
(49*a2*a4*a5*a7*a7)/1920. + (5*a4*a4*a5*a7*a7)/3168. + \
(1151*a3*a5*a5*a7*a7)/887040. + (25*a5*a5*a5*a7*a7)/576576. + \
(167*a2*a3*a6*a7*a7)/756. + (1129*a3*a4*a6*a7*a7)/26880. + \
(7331*a2*a5*a6*a7*a7)/181440. + (3359*a4*a5*a6*a7*a7)/591360. + \
(14143*a3*a6*a6*a7*a7)/483840. + (841*a5*a6*a6*a7*a7)/177408. + \
(1129*a2*a3*a7*a7*a7)/80640. + (2309*a3*a4*a7*a7*a7)/1.064448e6 + \
(3359*a2*a5*a7*a7*a7)/1.77408e6 + (25*a4*a5*a7*a7*a7)/109824. + \
(1999*a3*a6*a7*a7*a7)/591360. + (45137*a5*a6*a7*a7*a7)/1.05670656e8 + \
(31907*a3*a7*a7*a7*a7)/2.58306048e8 + (a5*a7*a7*a7*a7)/79872. + \
(7*a2*a2*a3*a3*a8)/8. + (a3*a3*a3*a3*a8)/48. + 4*a2*a2*a2*a4*a8 + \
(25*a2*a3*a3*a4*a8)/24. + (5*a2*a2*a4*a4*a8)/3. + \
(11*a3*a3*a4*a4*a8)/64. + (a2*a4*a4*a4*a8)/4. + \
(5*a4*a4*a4*a4*a8)/448. + (41*a2*a2*a3*a5*a8)/24. + \
(11*a3*a3*a3*a5*a8)/192. + (59*a2*a3*a4*a5*a8)/72. + \
(1013*a3*a4*a4*a5*a8)/12096. + (73*a2*a2*a5*a5*a8)/288. + \
(19*a3*a3*a5*a5*a8)/756. + (251*a2*a4*a5*a5*a8)/3024. + \
(23*a4*a4*a5*a5*a8)/3780. + (89*a3*a5*a5*a5*a8)/30240. + \
(541*a5*a5*a5*a5*a8)/5.98752e6 + (10*a2*a2*a2*a6*a8)/3. + \
(13*a2*a3*a3*a6*a8)/12. + (15*a2*a2*a4*a6*a8)/4. + \
(19*a3*a3*a4*a6*a8)/48. + (23*a2*a4*a4*a6*a8)/24. + \
(1153*a4*a4*a4*a6*a8)/18144. + (37*a2*a3*a5*a6*a8)/36. + \
(1391*a3*a4*a5*a6*a8)/6048. + (185*a2*a5*a5*a6*a8)/1512. + \
(3631*a4*a5*a5*a6*a8)/181440. + 2*a2*a2*a6*a6*a8 + \
(43*a3*a3*a6*a6*a8)/192. + (7*a2*a4*a6*a6*a8)/6. + \
(29*a4*a4*a6*a6*a8)/224. + (613*a3*a5*a6*a6*a8)/4032. + \
(3203*a5*a5*a6*a6*a8)/207360. + (11*a2*a6*a6*a6*a8)/24. + \
(683*a4*a6*a6*a6*a8)/6048. + (1301*a6*a6*a6*a6*a8)/36288. + \
(5*a2*a2*a2*a7*a8)/4. + (19*a2*a3*a3*a7*a8)/48. + \
(23*a2*a2*a4*a7*a8)/24. + (1231*a3*a3*a4*a7*a8)/12096. + \
(1153*a2*a4*a4*a7*a8)/6048. + (59*a4*a4*a4*a7*a8)/5760. + \
(1391*a2*a3*a5*a7*a8)/6048. + (15173*a3*a4*a5*a7*a8)/362880. + \
(3631*a2*a5*a5*a7*a8)/181440. + (89*a4*a5*a5*a7*a8)/31680. + \
(7*a2*a2*a6*a7*a8)/6. + (3085*a3*a3*a6*a7*a8)/24192. + \
(29*a2*a4*a6*a7*a8)/56. + (313*a4*a4*a6*a7*a8)/6720. + \
(11057*a3*a5*a6*a7*a8)/181440. + (114881*a5*a5*a6*a7*a8)/2.395008e7 + \
(683*a2*a6*a6*a7*a8)/2016. + (48911*a4*a6*a6*a7*a8)/725760. + \
(631*a6*a6*a6*a7*a8)/20160. + (29*a2*a2*a7*a7*a8)/224. + \
(8273*a3*a3*a7*a7*a8)/580608. + (313*a2*a4*a7*a7*a8)/6720. + \
(69*a4*a4*a7*a7*a8)/19712. + (18217*a3*a5*a7*a7*a8)/3.54816e6 + \
(19627*a5*a5*a7*a7*a8)/6.054048e7 + (48911*a2*a6*a7*a7*a8)/725760. + \
(19933*a4*a6*a7*a7*a8)/1.77408e6 + (15241*a6*a6*a7*a7*a8)/1.77408e6 + \
(19933*a2*a7*a7*a7*a8)/5.32224e6 + (6269*a4*a7*a7*a7*a8)/1.1741184e7 \
+ (1452571*a6*a7*a7*a7*a8)/1.6144128e9 + \
(349*a7*a7*a7*a7*a8)/1.1354112e7 + (7*a2*a2*a3*a8*a8)/8. + \
(19*a3*a3*a3*a8*a8)/576. + (37*a2*a3*a4*a8*a8)/72. + \
(187*a3*a4*a4*a8*a8)/3024. + (49*a2*a2*a5*a8*a8)/144. + \
(289*a3*a3*a5*a8*a8)/8064. + (15*a2*a4*a5*a8*a8)/112. + \
(2113*a4*a4*a5*a8*a8)/181440. + (1637*a3*a5*a5*a8*a8)/207360. + \
(2413*a5*a5*a5*a8*a8)/5.98752e6 + (89*a2*a3*a6*a8*a8)/144. + \
(9*a3*a4*a6*a8*a8)/56. + (559*a2*a5*a6*a8*a8)/3024. + \
(3643*a4*a5*a6*a8*a8)/103680. + (1231*a3*a6*a6*a8*a8)/12096. + \
(341*a5*a6*a6*a8*a8)/13440. + (9*a2*a3*a7*a8*a8)/56. + \
(24529*a3*a4*a7*a8*a8)/725760. + (3643*a2*a5*a7*a8*a8)/103680. + \
(5077*a4*a5*a7*a8*a8)/887040. + (13519*a3*a6*a7*a8*a8)/290304. + \
(862993*a5*a6*a7*a8*a8)/9.580032e7 + (15901*a3*a7*a7*a8*a8)/3.54816e6 \
+ (187399*a5*a7*a7*a8*a8)/2.690688e8 + (41*a2*a2*a8*a8*a8)/288. + \
(43*a3*a3*a8*a8*a8)/2688. + (11*a2*a4*a8*a8*a8)/168. + \
(1361*a4*a4*a8*a8*a8)/207360. + (2465*a3*a5*a8*a8*a8)/290304. + \
(75293*a5*a5*a8*a8*a8)/9.580032e7 + (29*a2*a6*a8*a8*a8)/336. + \
(47*a4*a6*a8*a8*a8)/2520. + (23*a6*a6*a8*a8*a8)/1792. + \
(47*a2*a7*a8*a8*a8)/2520. + (17*a4*a7*a8*a8*a8)/4928. + \
(490277*a6*a7*a8*a8*a8)/9.580032e7 + \
(87047*a7*a7*a8*a8*a8)/1.956864e8 + (257*a3*a8*a8*a8*a8)/80640. + \
(66571*a5*a8*a8*a8*a8)/9.580032e7 + (9*a8*a8*a8*a8*a8)/39424. + \
(5*a2*a2*a3*a3*a3*pi2)/3. + (5*a3*a3*a3*a3*a3*pi2)/144. + \
(25*a2*a2*a2*a3*a4*pi2)/3. + (5*a2*a3*a3*a3*a4*pi2)/4. + \
(25*a2*a2*a3*a4*a4*pi2)/12. + (7*a3*a3*a3*a4*a4*pi2)/72. + \
(a2*a3*a4*a4*a4*pi2)/12. + (10*a2*a2*a2*a2*a5*pi2)/3. + \
(25*a2*a2*a3*a3*a5*pi2)/12. + (7*a3*a3*a3*a3*a5*pi2)/144. + \
(5*a2*a2*a2*a4*a5*pi2)/6. + (31*a2*a3*a3*a4*a5*pi2)/72. + \
(a2*a2*a4*a4*a5*pi2)/18. + (a3*a3*a4*a4*a5*pi2)/84. + \
(5*a2*a2*a3*a5*a5*pi2)/36. + (25*a3*a3*a3*a5*a5*pi2)/3024. + \
(11*a2*a3*a4*a5*a5*pi2)/756. + (a2*a2*a5*a5*a5*pi2)/1134. + \
(a3*a3*a5*a5*a5*pi2)/3402. + 10*a2*a2*a2*a3*a6*pi2 + \
(125*a2*a3*a3*a3*a6*pi2)/72. + (15*a2*a2*a3*a4*a6*pi2)/2. + \
(13*a3*a3*a3*a4*a6*pi2)/36. + (5*a2*a3*a4*a4*a6*pi2)/6. + \
(5*a3*a4*a4*a4*a6*pi2)/336. + (20*a2*a2*a2*a5*a6*pi2)/9. + \
(71*a2*a3*a3*a5*a6*pi2)/72. + (13*a2*a2*a4*a5*a6*pi2)/24. + \
(187*a3*a3*a4*a5*a6*pi2)/2016. + (a2*a4*a4*a5*a6*pi2)/42. + \
(97*a2*a3*a5*a5*a6*pi2)/1512. + (a3*a4*a5*a5*a6*pi2)/336. + \
(a2*a5*a5*a5*a6*pi2)/1701. + (35*a2*a2*a3*a6*a6*pi2)/6. + \
(85*a3*a3*a3*a6*a6*pi2)/288. + (89*a2*a3*a4*a6*a6*pi2)/48. + \
(39*a3*a4*a4*a6*a6*pi2)/448. + (3*a2*a2*a5*a6*a6*pi2)/4. + \
(1427*a3*a3*a5*a6*a6*pi2)/12096. + (235*a2*a4*a5*a6*a6*pi2)/2016. + \
(5*a4*a4*a5*a6*a6*pi2)/2016. + (433*a3*a5*a5*a6*a6*pi2)/60480. + \
(13*a5*a5*a5*a6*a6*pi2)/199584. + (41*a2*a3*a6*a6*a6*pi2)/36. + \
(1339*a3*a4*a6*a6*a6*pi2)/9072. + (131*a2*a5*a6*a6*a6*pi2)/1134. + \
(49*a4*a5*a6*a6*a6*pi2)/5760. + (11*a3*a6*a6*a6*a6*pi2)/144. + \
(7331*a5*a6*a6*a6*a6*pi2)/1.08864e6 + (5*a2*a2*a2*a3*a7*pi2)/2. + \
(13*a2*a3*a3*a3*a7*pi2)/36. + (5*a2*a2*a3*a4*a7*pi2)/6. + \
(29*a3*a3*a3*a4*a7*pi2)/672. + (5*a2*a3*a4*a4*a7*pi2)/112. + \
(13*a2*a2*a2*a5*a7*pi2)/72. + (187*a2*a3*a3*a5*a7*pi2)/2016. + \
(a2*a2*a4*a5*a7*pi2)/42. + (5*a3*a3*a4*a5*a7*pi2)/1008. + \
(a2*a3*a5*a5*a7*pi2)/336. + (89*a2*a2*a3*a6*a7*pi2)/48. + \
(3385*a3*a3*a3*a6*a7*pi2)/36288. + (39*a2*a3*a4*a6*a7*pi2)/112. + \
(5*a3*a4*a4*a6*a7*pi2)/576. + (235*a2*a2*a5*a6*a7*pi2)/2016. + \
(829*a3*a3*a5*a6*a7*pi2)/40320. + (5*a2*a4*a5*a6*a7*pi2)/504. + \
(85*a3*a5*a5*a6*a7*pi2)/133056. + (1339*a2*a3*a6*a6*a7*pi2)/3024. + \
(757*a3*a4*a6*a6*a7*pi2)/20160. + (49*a2*a5*a6*a6*a7*pi2)/1920. + \
(5*a4*a5*a6*a6*a7*pi2)/4752. + (403*a3*a6*a6*a6*a7*pi2)/11520. + \
(3359*a5*a6*a6*a6*a7*pi2)/1.77408e6 + (39*a2*a2*a3*a7*a7*pi2)/448. + \
(97*a3*a3*a3*a7*a7*pi2)/20160. + (5*a2*a3*a4*a7*a7*pi2)/576. + \
(5*a2*a2*a5*a7*a7*pi2)/2016. + (5*a3*a3*a5*a7*a7*pi2)/9504. + \
(757*a2*a3*a6*a7*a7*pi2)/20160. + (5*a3*a4*a6*a7*a7*pi2)/2816. + \
(5*a2*a5*a6*a7*a7*pi2)/4752. + (2939*a3*a6*a6*a7*a7*pi2)/709632. + \
(25*a5*a6*a6*a7*a7*pi2)/219648. + (5*a2*a3*a7*a7*a7*pi2)/8448. + \
(5*a3*a6*a7*a7*a7*pi2)/39936. + (25*a2*a2*a2*a2*a8*pi2)/6. + \
(10*a2*a2*a3*a3*a8*pi2)/3. + (49*a3*a3*a3*a3*a8*pi2)/576. + \
(5*a2*a2*a2*a4*a8*pi2)/2. + (167*a2*a3*a3*a4*a8*pi2)/144. + \
(59*a2*a2*a4*a4*a8*pi2)/144. + (11*a3*a3*a4*a4*a8*pi2)/168. + \
(5*a2*a4*a4*a4*a8*pi2)/336. + (49*a2*a2*a3*a5*a8*pi2)/48. + \
(1915*a3*a3*a3*a5*a8*pi2)/36288. + (577*a2*a3*a4*a5*a8*pi2)/3024. + \
(5*a3*a4*a4*a5*a8*pi2)/1008. + (47*a2*a2*a5*a5*a8*pi2)/1512. + \
(527*a3*a3*a5*a5*a8*pi2)/90720. + (a2*a4*a5*a5*a8*pi2)/336. + \
(13*a3*a5*a5*a5*a8*pi2)/99792. + (40*a2*a2*a2*a6*a8*pi2)/9. + \
(587*a2*a3*a3*a6*a8*pi2)/288. + (23*a2*a2*a4*a6*a8*pi2)/12. + \
(3589*a3*a3*a4*a6*a8*pi2)/12096. + (59*a2*a4*a4*a6*a8*pi2)/336. + \
(5*a4*a4*a4*a6*a8*pi2)/1728. + (111*a2*a3*a5*a6*a8*pi2)/224. + \
(283*a3*a4*a5*a6*a8*pi2)/6720. + (331*a2*a5*a5*a6*a8*pi2)/22680. + \
(85*a4*a5*a5*a6*a8*pi2)/133056. + (15*a2*a2*a6*a6*a8*pi2)/8. + \
(6961*a3*a3*a6*a6*a8*pi2)/24192. + (2789*a2*a4*a6*a6*a8*pi2)/6048. + \
(171*a4*a4*a6*a6*a8*pi2)/8960. + (6151*a3*a5*a6*a6*a8*pi2)/103680. + \
(733*a5*a5*a6*a6*a8*pi2)/443520. + (1501*a2*a6*a6*a6*a8*pi2)/4536. + \
(23*a4*a6*a6*a6*a8*pi2)/630. + (18485*a6*a6*a6*a6*a8*pi2)/870912. + \
(23*a2*a2*a2*a7*a8*pi2)/36. + (3589*a2*a3*a3*a7*a8*pi2)/12096. + \
(59*a2*a2*a4*a7*a8*pi2)/336. + (1189*a3*a3*a4*a7*a8*pi2)/40320. + \
(5*a2*a4*a4*a7*a8*pi2)/576. + (283*a2*a3*a5*a7*a8*pi2)/6720. + \
(5*a3*a4*a5*a7*a8*pi2)/2376. + (85*a2*a5*a5*a7*a8*pi2)/133056. + \
(2789*a2*a2*a6*a7*a8*pi2)/6048. + (847*a3*a3*a6*a7*a8*pi2)/11520. + \
(171*a2*a4*a6*a7*a8*pi2)/2240. + (5*a4*a4*a6*a7*a8*pi2)/2816. + \
(25231*a3*a5*a6*a7*a8*pi2)/2.66112e6 + \
(415*a5*a5*a6*a7*a8*pi2)/2.965248e6 + (23*a2*a6*a6*a7*a8*pi2)/210. + \
(281*a4*a6*a6*a7*a8*pi2)/33264. + \
(15353*a6*a6*a6*a7*a8*pi2)/1.77408e6 + (171*a2*a2*a7*a7*a8*pi2)/8960. \
+ (3551*a3*a3*a7*a7*a8*pi2)/1.064448e6 + (5*a2*a4*a7*a7*a8*pi2)/2816. \
+ (25*a3*a5*a7*a7*a8*pi2)/109824. + (281*a2*a6*a7*a7*a8*pi2)/33264. + \
(5*a4*a6*a7*a7*a8*pi2)/13312. + \
(27233*a6*a6*a7*a7*a8*pi2)/2.8700672e7 + \
(5*a2*a7*a7*a7*a8*pi2)/39936. + (a6*a7*a7*a7*a8*pi2)/36864. + \
(83*a2*a2*a3*a8*a8*pi2)/72. + (8569*a3*a3*a3*a8*a8*pi2)/145152. + \
(53*a2*a3*a4*a8*a8*pi2)/168. + (607*a3*a4*a4*a8*a8*pi2)/40320. + \
(775*a2*a2*a5*a8*a8*pi2)/6048. + (241*a3*a3*a5*a8*a8*pi2)/11520. + \
(869*a2*a4*a5*a8*a8*pi2)/40320. + (5*a4*a4*a5*a8*a8*pi2)/9504. + \
(3643*a3*a5*a5*a8*a8*pi2)/2.66112e6 + \
(25*a5*a5*a5*a8*a8*pi2)/1.729728e6 + (275*a2*a3*a6*a8*a8*pi2)/432. + \
(6263*a3*a4*a6*a8*a8*pi2)/80640. + (179*a2*a5*a6*a8*a8*pi2)/2880. + \
(25831*a4*a5*a6*a8*a8*pi2)/5.32224e6 + \
(2269*a3*a6*a6*a8*a8*pi2)/26880. + \
(39581*a5*a6*a6*a8*a8*pi2)/5.32224e6 + \
(6263*a2*a3*a7*a8*a8*pi2)/80640. + \
(7277*a3*a4*a7*a8*a8*pi2)/1.064448e6 + \
(25831*a2*a5*a7*a8*a8*pi2)/5.32224e6 + \
(25*a4*a5*a7*a8*a8*pi2)/109824. + \
(100693*a3*a6*a7*a8*a8*pi2)/5.32224e6 + \
(1285681*a5*a6*a7*a8*a8*pi2)/1.162377216e9 + \
(16811*a3*a7*a7*a8*a8*pi2)/2.1525504e7 + (a5*a7*a7*a8*a8*pi2)/39936. \
+ (13*a2*a2*a8*a8*a8*pi2)/112. + (3523*a3*a3*a8*a8*a8*pi2)/193536. + \
(551*a2*a4*a8*a8*a8*pi2)/20160. + (23*a4*a4*a8*a8*a8*pi2)/19712. + \
(7705*a3*a5*a8*a8*a8*pi2)/2.128896e6 + \
(19627*a5*a5*a8*a8*a8*pi2)/1.8162144e8 + \
(185*a2*a6*a8*a8*a8*pi2)/3024. + (35323*a4*a6*a8*a8*a8*pi2)/5.32224e6 \
+ (41897*a6*a6*a8*a8*a8*pi2)/5.32224e6 + \
(35323*a2*a7*a8*a8*a8*pi2)/5.32224e6 + \
(6269*a4*a7*a8*a8*a8*pi2)/1.1741184e7 + \
(323899*a6*a7*a8*a8*a8*pi2)/2.018016e8 + \
(349*a7*a7*a8*a8*a8*pi2)/5.677056e6 + \
(28141*a3*a8*a8*a8*a8*pi2)/1.064448e7 + \
(187399*a5*a8*a8*a8*a8*pi2)/8.072064e8 + \
(87047*a8*a8*a8*a8*a8*pi2)/5.870592e8 + (14*a2*a2*a2*a2*a3*pi4)/3. + \
(49*a2*a2*a3*a3*a3*pi4)/36. + (49*a3*a3*a3*a3*a3*pi4)/2160. + \
(7*a2*a2*a2*a3*a4*pi4)/9. + (7*a2*a3*a3*a3*a4*pi4)/60. + \
(7*a2*a2*a3*a3*a5*pi4)/90. + (a3*a3*a3*a3*a5*pi4)/360. + \
(7*a2*a2*a2*a3*a6*pi4)/2. + (77*a2*a3*a3*a3*a6*pi4)/135. + \
(7*a2*a2*a3*a4*a6*pi4)/20. + (a3*a3*a3*a4*a6*pi4)/48. + \
(7*a2*a2*a2*a5*a6*pi4)/135. + (a2*a3*a3*a5*a6*pi4)/30. + \
(371*a2*a2*a3*a6*a6*pi4)/360. + (35*a3*a3*a3*a6*a6*pi4)/576. + \
(a2*a3*a4*a6*a6*pi4)/16. + (a2*a2*a5*a6*a6*pi4)/60. + \
(a3*a3*a5*a6*a6*pi4)/288. + (41*a2*a3*a6*a6*a6*pi4)/288. + \
(7*a3*a4*a6*a6*a6*pi4)/1728. + (a2*a5*a6*a6*a6*pi4)/432. + \
(11*a3*a6*a6*a6*a6*pi4)/1440. + (7*a5*a6*a6*a6*a6*pi4)/57024. + \
(7*a2*a2*a2*a3*a7*pi4)/60. + (a2*a3*a3*a3*a7*pi4)/48. + \
(a2*a2*a3*a6*a7*pi4)/16. + (7*a3*a3*a3*a6*a7*pi4)/1728. + \
(7*a2*a3*a6*a6*a7*pi4)/576. + (7*a3*a6*a6*a6*a7*pi4)/8448. + \
(7*a2*a2*a2*a2*a8*pi4)/9. + (203*a2*a2*a3*a3*a8*pi4)/240. + \
(73*a3*a3*a3*a3*a8*pi4)/2880. + (7*a2*a2*a2*a4*a8*pi4)/60. + \
(a2*a3*a3*a4*a8*pi4)/16. + (a2*a2*a3*a5*a8*pi4)/30. + \
(a3*a3*a3*a5*a8*pi4)/432. + (49*a2*a2*a2*a6*a8*pi4)/72. + \
(11*a2*a3*a3*a6*a8*pi4)/30. + (a2*a2*a4*a6*a8*pi4)/16. + \
(7*a3*a3*a4*a6*a8*pi4)/576. + (a2*a3*a5*a6*a8*pi4)/72. + \
(103*a2*a2*a6*a6*a8*pi4)/480. + (463*a3*a3*a6*a6*a8*pi4)/11520. + \
(7*a2*a4*a6*a6*a8*pi4)/576. + (7*a3*a5*a6*a6*a8*pi4)/4752. + \
(533*a2*a6*a6*a6*a8*pi4)/17280. + (7*a4*a6*a6*a6*a8*pi4)/8448. + \
(775*a6*a6*a6*a6*a8*pi4)/456192. + (a2*a2*a2*a7*a8*pi4)/48. + \
(7*a2*a3*a3*a7*a8*pi4)/576. + (7*a2*a2*a6*a7*a8*pi4)/576. + \
(7*a3*a3*a6*a7*a8*pi4)/2816. + (7*a2*a6*a6*a7*a8*pi4)/2816. + \
(7*a6*a6*a6*a7*a8*pi4)/39936. + (59*a2*a2*a3*a8*a8*pi4)/320. + \
(199*a3*a3*a3*a8*a8*pi4)/17280. + (7*a2*a3*a4*a8*a8*pi4)/576. + \
(a2*a2*a5*a8*a8*pi4)/288. + (7*a3*a3*a5*a8*a8*pi4)/9504. + \
(13*a2*a3*a6*a8*a8*pi4)/160. + (7*a3*a4*a6*a8*a8*pi4)/2816. + \
(7*a2*a5*a6*a8*a8*pi4)/4752. + (919*a3*a6*a6*a8*a8*pi4)/101376. + \
(35*a5*a6*a6*a8*a8*pi4)/219648. + (7*a2*a3*a7*a8*a8*pi4)/2816. + \
(7*a3*a6*a7*a8*a8*pi4)/13312. + (473*a2*a2*a8*a8*a8*pi4)/34560. + \
(1207*a3*a3*a8*a8*a8*pi4)/456192. + (7*a2*a4*a8*a8*a8*pi4)/8448. + \
(35*a3*a5*a8*a8*a8*pi4)/329472. + (349*a2*a6*a8*a8*a8*pi4)/57024. + \
(7*a4*a6*a8*a8*a8*pi4)/39936. + \
(76603*a6*a6*a8*a8*a8*pi4)/1.10702592e8 + \
(7*a2*a7*a8*a8*a8*pi4)/39936. + (7*a6*a7*a8*a8*a8*pi4)/184320. + \
(33965*a3*a8*a8*a8*a8*pi4)/1.10702592e8 + \
(7*a5*a8*a8*a8*a8*pi4)/1.19808e6 + \
(349*a8*a8*a8*a8*a8*pi4)/2.433024e7;

    Float m6 = a3*a3*a3*a3*a3*a3/448. + (3*a2*a3*a3*a3*a3*a4)/16. + \
3*a2*a2*a3*a3*a4*a4 + (5*a3*a3*a3*a3*a4*a4)/64. + 5*a2*a2*a2*a4*a4*a4 \
+ (5*a2*a3*a3*a4*a4*a4)/4. + (5*a2*a2*a4*a4*a4*a4)/4. + \
(19*a3*a3*a4*a4*a4*a4)/192. + (23*a2*a4*a4*a4*a4*a4)/240. + \
a4*a4*a4*a4*a4*a4/448. + (3*a2*a2*a3*a3*a3*a5)/4. + \
(a3*a3*a3*a3*a3*a5)/64. + 9*a2*a2*a2*a3*a4*a5 + \
(9*a2*a3*a3*a3*a4*a5)/8. + (25*a2*a2*a3*a4*a4*a5)/4. + \
(19*a3*a3*a3*a4*a4*a5)/96. + (23*a2*a3*a4*a4*a4*a5)/24. + \
(163*a3*a4*a4*a4*a4*a5)/4032. + 3*a2*a2*a2*a2*a5*a5 + \
(11*a2*a2*a3*a3*a5*a5)/8. + (5*a3*a3*a3*a3*a5*a5)/192. + \
(5*a2*a2*a2*a4*a5*a5)/2. + (37*a2*a3*a3*a4*a5*a5)/48. + \
(17*a2*a2*a4*a4*a5*a5)/24. + (19*a3*a3*a4*a4*a5*a5)/252. + \
(a2*a4*a4*a4*a5*a5)/14. + (a4*a4*a4*a4*a5*a5)/448. + \
(13*a2*a2*a3*a5*a5*a5)/48. + (127*a3*a3*a3*a5*a5*a5)/12096. + \
(25*a2*a3*a4*a5*a5*a5)/252. + (109*a3*a4*a4*a5*a5*a5)/15120. + \
(11*a2*a2*a5*a5*a5*a5)/1008. + (a3*a3*a5*a5*a5*a5)/720. + \
(47*a2*a4*a5*a5*a5*a5)/15120. + (73*a4*a4*a5*a5*a5*a5)/399168. + \
(139*a3*a5*a5*a5*a5*a5)/1.99584e6 + \
(53*a5*a5*a5*a5*a5*a5)/4.6702656e7 + (9*a2*a3*a3*a3*a3*a6)/56. + \
(21*a2*a2*a3*a3*a4*a6)/4. + (5*a3*a3*a3*a3*a4*a6)/32. + \
12*a2*a2*a2*a4*a4*a6 + (15*a2*a3*a3*a4*a4*a6)/4. + \
5*a2*a2*a4*a4*a4*a6 + (43*a3*a3*a4*a4*a4*a6)/96. + \
(7*a2*a4*a4*a4*a4*a6)/12. + (11*a4*a4*a4*a4*a4*a6)/560. + \
8*a2*a2*a2*a3*a5*a6 + (9*a2*a3*a3*a3*a5*a6)/8. + \
(51*a2*a2*a3*a4*a5*a6)/4. + (43*a3*a3*a3*a4*a5*a6)/96. + \
(7*a2*a3*a4*a4*a5*a6)/2. + (1391*a3*a4*a4*a4*a5*a6)/6048. + \
3*a2*a2*a2*a5*a5*a6 + (11*a2*a3*a3*a5*a5*a6)/12. + \
(31*a2*a2*a4*a5*a5*a6)/16. + (13*a3*a3*a4*a5*a5*a6)/63. + \
(335*a2*a4*a4*a5*a5*a6)/1008. + (239*a4*a4*a4*a5*a5*a6)/15120. + \
(97*a2*a3*a5*a5*a5*a6)/672. + (17*a3*a4*a5*a5*a5*a6)/720. + \
(163*a2*a5*a5*a5*a5*a6)/30240. + (353*a4*a5*a5*a5*a5*a6)/498960. + \
(33*a2*a2*a3*a3*a6*a6)/14. + (5*a3*a3*a3*a3*a6*a6)/64. + \
10*a2*a2*a2*a4*a6*a6 + (15*a2*a3*a3*a4*a6*a6)/4. + \
(15*a2*a2*a4*a4*a6*a6)/2. + (3*a3*a3*a4*a4*a6*a6)/4. + \
(11*a2*a4*a4*a4*a6*a6)/8. + (409*a4*a4*a4*a4*a6*a6)/6048. + \
(13*a2*a2*a3*a5*a6*a6)/2. + (a3*a3*a3*a5*a6*a6)/4. + \
(33*a2*a3*a4*a5*a6*a6)/8. + (937*a3*a4*a4*a5*a6*a6)/2016. + \
(5*a2*a2*a5*a5*a6*a6)/4. + (1091*a3*a3*a5*a5*a6*a6)/8064. + \
(163*a2*a4*a5*a5*a6*a6)/336. + (673*a4*a4*a5*a5*a6*a6)/17280. + \
(4423*a3*a5*a5*a5*a6*a6)/241920. + (5101*a5*a5*a5*a5*a6*a6)/7.98336e6 \
+ (20*a2*a2*a2*a6*a6*a6)/7. + (5*a2*a3*a3*a6*a6*a6)/4. + \
5*a2*a2*a4*a6*a6*a6 + (53*a3*a3*a4*a6*a6*a6)/96. + \
(19*a2*a4*a4*a6*a6*a6)/12. + (359*a4*a4*a4*a6*a6*a6)/3024. + \
(19*a2*a3*a5*a6*a6*a6)/12. + (809*a3*a4*a5*a6*a6*a6)/2016. + \
(113*a2*a5*a5*a6*a6*a6)/504. + (9707*a4*a5*a5*a6*a6*a6)/241920. + \
(5*a2*a2*a6*a6*a6*a6)/4. + (29*a3*a3*a6*a6*a6*a6)/192. + \
(43*a2*a4*a6*a6*a6*a6)/48. + (457*a4*a4*a6*a6*a6*a6)/4032. + \
(1525*a3*a5*a6*a6*a6*a6)/12096. + (3569*a5*a5*a6*a6*a6*a6)/241920. + \
(a2*a6*a6*a6*a6*a6)/5. + (85*a4*a6*a6*a6*a6*a6)/1512. + \
(43*a6*a6*a6*a6*a6*a6)/3780. + (25*a2*a2*a3*a3*a3*a3*pi2)/16. + \
(5*a3*a3*a3*a3*a3*a3*pi2)/192. + (35*a2*a2*a2*a3*a3*a4*pi2)/2. + \
(5*a2*a3*a3*a3*a3*a4*pi2)/3. + 10*a2*a2*a2*a2*a4*a4*pi2 + \
(65*a2*a2*a3*a3*a4*a4*pi2)/8. + (19*a3*a3*a3*a3*a4*a4*pi2)/96. + \
(5*a2*a2*a2*a4*a4*a4*pi2)/3. + (17*a2*a3*a3*a4*a4*a4*pi2)/24. + \
(a2*a2*a4*a4*a4*a4*pi2)/16. + (5*a3*a3*a4*a4*a4*a4*pi2)/448. + \
15*a2*a2*a2*a2*a3*a5*pi2 + (35*a2*a2*a3*a3*a3*a5*pi2)/8. + \
(19*a3*a3*a3*a3*a3*a5*pi2)/288. + (65*a2*a2*a2*a3*a4*a5*pi2)/6. + \
(37*a2*a3*a3*a3*a4*a5*pi2)/24. + (35*a2*a2*a3*a4*a4*a5*pi2)/24. + \
(163*a3*a3*a3*a4*a4*a5*pi2)/2016. + (a2*a3*a4*a4*a4*a5*pi2)/21. + \
(5*a2*a2*a2*a2*a5*a5*pi2)/6. + (23*a2*a2*a3*a3*a5*a5*pi2)/24. + \
(19*a3*a3*a3*a3*a5*a5*pi2)/756. + (a2*a2*a2*a4*a5*a5*pi2)/3. + \
(43*a2*a3*a3*a4*a5*a5*pi2)/252. + (11*a2*a2*a4*a4*a5*a5*pi2)/504. + \
(a3*a3*a4*a4*a5*a5*pi2)/224. + (31*a2*a2*a3*a5*a5*a5*pi2)/756. + \
(109*a3*a3*a3*a5*a5*a5*pi2)/45360. + (2*a2*a3*a4*a5*a5*a5*pi2)/567. + \
(a2*a2*a5*a5*a5*a5*pi2)/3024. + (73*a3*a3*a5*a5*a5*a5*pi2)/1.197504e6 \
+ 20*a2*a2*a2*a3*a3*a6*pi2 + (205*a2*a3*a3*a3*a3*a6*pi2)/96. + \
25*a2*a2*a2*a2*a4*a6*pi2 + (95*a2*a2*a3*a3*a4*a6*pi2)/4. + \
(59*a3*a3*a3*a3*a4*a6*pi2)/96. + 10*a2*a2*a2*a4*a4*a6*pi2 + \
(69*a2*a3*a3*a4*a4*a6*pi2)/16. + (23*a2*a2*a4*a4*a4*a6*pi2)/24. + \
(17*a3*a3*a4*a4*a4*a6*pi2)/112. + (5*a2*a4*a4*a4*a4*a6*pi2)/224. + \
(55*a2*a2*a2*a3*a5*a6*pi2)/3. + (385*a2*a3*a3*a3*a5*a6*pi2)/144. + \
(59*a2*a2*a3*a4*a5*a6*pi2)/8. + (2293*a3*a3*a3*a4*a5*a6*pi2)/6048. + \
(211*a2*a3*a4*a4*a5*a6*pi2)/336. + (5*a3*a4*a4*a4*a5*a6*pi2)/504. + \
(17*a2*a2*a2*a5*a5*a6*pi2)/18. + (155*a2*a3*a3*a5*a5*a6*pi2)/336. + \
(3*a2*a2*a4*a5*a5*a6*pi2)/14. + (97*a3*a3*a4*a5*a5*a6*pi2)/2520. + \
(a2*a4*a4*a5*a5*a6*pi2)/112. + (407*a2*a3*a5*a5*a5*a6*pi2)/22680. + \
(13*a3*a4*a5*a5*a5*a6*pi2)/16632. + \
(73*a2*a5*a5*a5*a5*a6*pi2)/598752. + 15*a2*a2*a2*a2*a6*a6*pi2 + \
(65*a2*a2*a3*a3*a6*a6*pi2)/4. + (343*a3*a3*a3*a3*a6*a6*pi2)/768. + \
(50*a2*a2*a2*a4*a6*a6*pi2)/3. + (239*a2*a3*a3*a4*a6*a6*pi2)/32. + \
(7*a2*a2*a4*a4*a6*a6*pi2)/2. + (4303*a3*a3*a4*a4*a6*a6*pi2)/8064. + \
(11*a2*a4*a4*a4*a6*a6*pi2)/56. + (5*a4*a4*a4*a4*a6*a6*pi2)/2304. + \
(22*a2*a2*a3*a5*a6*a6*pi2)/3. + (8947*a3*a3*a3*a5*a6*a6*pi2)/24192. + \
(3487*a2*a3*a4*a5*a6*a6*pi2)/2016. + \
(929*a3*a4*a4*a5*a6*a6*pi2)/13440. + \
(335*a2*a2*a5*a5*a6*a6*pi2)/1008. + \
(6689*a3*a3*a5*a5*a6*a6*pi2)/120960. + \
(239*a2*a4*a5*a5*a6*a6*pi2)/5040. + (85*a4*a4*a5*a5*a6*a6*pi2)/88704. \
+ (4073*a3*a5*a5*a5*a6*a6*pi2)/1.99584e6 + \
(281*a5*a5*a5*a5*a6*a6*pi2)/2.0756736e7 + \
(25*a2*a2*a2*a6*a6*a6*pi2)/3. + (47*a2*a3*a3*a6*a6*a6*pi2)/12. + \
(55*a2*a2*a4*a6*a6*a6*pi2)/12. + (4205*a3*a3*a4*a6*a6*a6*pi2)/6048. + \
(409*a2*a4*a4*a6*a6*a6*pi2)/756. + (563*a4*a4*a4*a6*a6*a6*pi2)/40320. \
+ (91*a2*a3*a5*a6*a6*a6*pi2)/72. + \
(49387*a3*a4*a5*a6*a6*a6*pi2)/362880. + \
(673*a2*a5*a5*a6*a6*a6*pi2)/12960. + \
(197*a4*a5*a5*a6*a6*a6*pi2)/55440. + (95*a2*a2*a6*a6*a6*a6*pi2)/48. + \
(1837*a3*a3*a6*a6*a6*a6*pi2)/6048. + \
(1795*a2*a4*a6*a6*a6*a6*pi2)/3024. + \
(141*a4*a4*a6*a6*a6*a6*pi2)/4480. + \
(58223*a3*a5*a6*a6*a6*a6*pi2)/725760. + \
(146429*a5*a5*a6*a6*a6*a6*pi2)/4.790016e7 + \
(457*a2*a6*a6*a6*a6*a6*pi2)/2016. + \
(21757*a4*a6*a6*a6*a6*a6*pi2)/725760. + \
(3737*a6*a6*a6*a6*a6*a6*pi2)/362880. + (77*a2*a2*a2*a2*a3*a3*pi4)/4. \
+ (91*a2*a2*a3*a3*a3*a3*pi4)/24. + (133*a3*a3*a3*a3*a3*a3*pi4)/2880. \
+ 7*a2*a2*a2*a2*a2*a4*pi4 + (28*a2*a2*a2*a3*a3*a4*pi4)/3. + \
(553*a2*a3*a3*a3*a3*a4*pi4)/720. + (7*a2*a2*a2*a2*a4*a4*pi4)/12. + \
(21*a2*a2*a3*a3*a4*a4*pi4)/40. + (a3*a3*a3*a3*a4*a4*pi4)/64. + \
(35*a2*a2*a2*a2*a3*a5*pi4)/12. + (371*a2*a2*a3*a3*a3*a5*pi4)/360. + \
(163*a3*a3*a3*a3*a3*a5*pi4)/8640. + (14*a2*a2*a2*a3*a4*a5*pi4)/45. + \
(a2*a3*a3*a3*a4*a5*pi4)/15. + (7*a2*a2*a2*a2*a5*a5*pi4)/180. + \
(11*a2*a2*a3*a3*a5*a5*pi4)/360. + (a3*a3*a3*a3*a5*a5*pi4)/960. + \
14*a2*a2*a2*a2*a2*a6*pi4 + (287*a2*a2*a2*a3*a3*a6*pi4)/12. + \
(1519*a2*a3*a3*a3*a3*a6*pi4)/720. + (35*a2*a2*a2*a2*a4*a6*pi4)/6. + \
(679*a2*a2*a3*a3*a4*a6*pi4)/120. + (a3*a3*a3*a3*a4*a6*pi4)/6. + \
(7*a2*a2*a2*a4*a4*a6*pi4)/20. + (3*a2*a3*a3*a4*a4*a6*pi4)/16. + \
(427*a2*a2*a2*a3*a5*a6*pi4)/180. + (959*a2*a3*a3*a3*a5*a6*pi4)/2160. \
+ (a2*a2*a3*a4*a5*a6*pi4)/5. + (a3*a3*a3*a4*a5*a6*pi4)/72. + \
(11*a2*a2*a2*a5*a5*a6*pi4)/540. + (a2*a3*a3*a5*a5*a6*pi4)/80. + \
(35*a2*a2*a2*a2*a6*a6*pi4)/4. + (4333*a2*a2*a3*a3*a6*a6*pi4)/480. + \
(925*a3*a3*a3*a3*a6*a6*pi4)/3456. + (161*a2*a2*a2*a4*a6*a6*pi4)/72. + \
(19*a2*a3*a3*a4*a6*a6*pi4)/16. + (3*a2*a2*a4*a4*a6*a6*pi4)/32. + \
(7*a3*a3*a4*a4*a6*a6*pi4)/384. + (1103*a2*a2*a3*a5*a6*a6*pi4)/1440. + \
(283*a3*a3*a3*a5*a6*a6*pi4)/5760. + (a2*a3*a4*a5*a6*a6*pi4)/24. + \
(a2*a2*a5*a5*a6*a6*pi4)/160. + (17*a3*a3*a5*a5*a6*a6*pi4)/12672. + \
(49*a2*a2*a2*a6*a6*a6*pi4)/18. + (2507*a2*a3*a3*a6*a6*a6*pi4)/1728. + \
(11*a2*a2*a4*a6*a6*a6*pi4)/24. + (493*a3*a3*a4*a6*a6*a6*pi4)/5760. + \
(7*a2*a4*a4*a6*a6*a6*pi4)/576. + (323*a2*a3*a5*a6*a6*a6*pi4)/2880. + \
(7*a3*a4*a5*a6*a6*a6*pi4)/2376. + (17*a2*a5*a5*a6*a6*a6*pi4)/19008. + \
(409*a2*a2*a6*a6*a6*a6*pi4)/864. + (797*a3*a3*a6*a6*a6*a6*pi4)/9216. \
+ (563*a2*a4*a6*a6*a6*a6*pi4)/11520. + \
(7*a4*a4*a6*a6*a6*a6*pi4)/11264. + \
(9517*a3*a5*a6*a6*a6*a6*pi4)/1.52064e6 + \
(581*a5*a5*a6*a6*a6*a6*pi4)/1.1860992e7 + \
(141*a2*a6*a6*a6*a6*a6*pi4)/3200. + \
(1627*a4*a6*a6*a6*a6*a6*pi4)/760320. + \
(13063*a6*a6*a6*a6*a6*a6*pi4)/7.6032e6 + \
(31*a2*a2*a2*a2*a2*a2*pi6)/21. + (155*a2*a2*a2*a2*a3*a3*pi6)/84. + \
(31*a2*a2*a3*a3*a3*a3*pi6)/112. + (31*a3*a3*a3*a3*a3*a3*pi6)/9408. + \
(31*a2*a2*a2*a2*a2*a6*pi6)/42. + (31*a2*a2*a2*a3*a3*a6*pi6)/28. + \
(155*a2*a3*a3*a3*a3*a6*pi6)/1568. + (31*a2*a2*a2*a2*a6*a6*pi6)/112. + \
(465*a2*a2*a3*a3*a6*a6*pi6)/1568. + \
(155*a3*a3*a3*a3*a6*a6*pi6)/16128. + \
(155*a2*a2*a2*a6*a6*a6*pi6)/2352. + (155*a2*a3*a3*a6*a6*a6*pi6)/4032. \
+ (155*a2*a2*a6*a6*a6*a6*pi6)/16128. + \
(155*a3*a3*a6*a6*a6*a6*pi6)/78848. + \
(31*a2*a6*a6*a6*a6*a6*pi6)/39424. + \
(31*a6*a6*a6*a6*a6*a6*pi6)/1.118208e6;


        /* variance of empirical mean */
        Float w = var/N;
        std::cout << w << "\t";
        Float t1 = Float(0.5)*(m1-mean)*(m1-mean)/w;

        /* variance of empirical variance - estimated as var^2 / N */
        w *= var;
        Float t2 = Float(0.5)*(m2-var)*(m2-var)/w;

        w *= var*(2*2);
        Float t3 = Float(0.5)*m3*m3/w;

        w *= var*(3*3);
        Float k4 = m4 - 3*m2*m2;
        Float t4 = Float(0.5)*k4*k4/w;

        w *= var*(4*4);
        Float k5 = m5 - 10*m3*m2;
        Float t5 = Float(0.5)*k5*k5/w;

        w *= var*(5*5);
        Float k6 = m6 - 15*m4*m2 - 10*m3*m3 + 30*m2*m2*m2;
        Float t6 = Float(0.5)*k6*k6/w;

        loglike -= t1;
        loglike -= t2;
        loglike -= t3;
        loglike -= t4;
        loglike -= t5;
        loglike -= t6;

        /* gaussian prior around for quantities that have even number of terms, therefore for y->1-y they don't do x->-x, unlike the odd terms,
         * so the even terms are zero for all symmetric distributions, and we want something not too skewed. */
        Float t0 = 0;
        for (auto j : std::vector<size_t>{0, 2, 4, 7,8})
            t0 += 0.025*coords[0][j]*coords[0][j]*0.5/(var);

        // loglike -= t0;
        // for (auto j : std::vector<size_t>{1, 3, 5, 6})
        //     loglike -= coords[0][j]*coords[0][j]*0.5/(var);
        //
        std::cout << N << " " << loglike << ": mean=" << mean << " [" << m1 <<  " " << m2 << "(" << var << ") " << m3 << " " << m4 << " " << m5 << " " << m6 << "] " << t0 << " ( " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << " " << t6 << ")\n";

        /* compute central moments by integrating powers of quantile function. Do not integrate into tails too much for numerical stability.
         * TODO: explore better integration strategies - note they should be symmetric by construction to allow for cancellation */
        // Float dy = 0.0001;
        // Float Delta = 0.1;
        // Float m1 = 0;
        // for (Float y = Delta; y < 1-Delta+1e-7; y+=dy) {
        //     Float x = keelin_Q(y, coords[0]);
        //     m1 += x;
        // }
        // m1 *= dy;
        // Float mHigherThan1[5];
        // for (Float y = Delta; y < 1-Delta+1e-7; y+=dy) {
        //     Float x = keelin_Q(y, coords[0]) - m1;
        //
        //     for (int k = 0; k < 5; ++k) {
        //         x *= x;
        //         mHigherThan1[k] += x;
        //     }
        // }
        // for (int k = 0; k < 5; ++k) {
        //     mHigherThan1[k] *= dy;
        // }
        //
        // Float w = var;
        /* first moment - mean of keelin - should be near zero, within std */
        // loglike -= Float(0.5)*m1*m1/w;
        /* second central moment - var of keelin - should be near empirical var */
        // w *= var;
        // loglike -= Float(0.5)*(mHigherThan1[0]-var)*(mHigherThan1[0]-var)/w;
        /* third central moment - skew - should be near 0 */
        // w *= var;
        // loglike -= Float(0.1)*Float(0.5)*mHigherThan1[1]*mHigherThan1[1]/w;
        /* fourth cumulant (=4th central moment - 3 var^2 ) should be near zero, as for the std normal which has 4th central moment = 3*/
        // w *= var;
        // Float k4 = mHigherThan1[2] - 3*mHigherThan1[0]*mHigherThan1[0];
        // loglike -= Float(1)*Float(0.5)*k4*k4/w;
        /* higher cumulants should be near zero too with a std normal prior */
        // w *= var;
        // Float k5 = mHigherThan1[3] - 10*mHigherThan1[1]*mHigherThan1[0];
        // loglike -= Float(0.01)*Float(0.5)*k5*k5/w;
        // w *= var;
        // Float k6 = mHigherThan1[4] - 15*mHigherThan1[2]*mHigherThan1[0] - 10*mHigherThan1[1]*mHigherThan1[1] + 30*mHigherThan1[0]*mHigherThan1[0]*mHigherThan1[0];
        // loglike -= Float(0.01)*Float(0.5)*k6*k6/w;

    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = std::dynamic_pointer_cast<KeelinPDF>(copy());
        // newstate->eval(shared);
        // return Proposal{newstate, 1};

        int nSteps = rnd.nextFloat() * 10;

        for (int k = 0; k < nSteps; ++k) {
            int idx = std::min(int(rnd.nextFloat()*nTerms), nTerms-1);
            int idx2 = std::min(int(rnd.nextFloat()*2), 1);

            newstate->coords[idx2][idx] += (rnd.nextFloat()-0.5)*std/std::sqrt(N)*std::min(stepsizeCorrectionFac, Float(1));
        }

                        // bound(newstate->coords[2][whichMode], minSigma, maxSigma);

        newstate->eval(shared);
        return Proposal{newstate, 1};

    }

    std::shared_ptr<SubspaceState> copy() const override {
        auto foo = new KeelinPDF(*this);
        return std::shared_ptr<SubspaceState>(foo);
    }
private:

    const ProbabilityDistributionSamples* data;

};
#endif



class GaussKeelinMixturePDF : public SubspaceState {

private:

    SumConstraint constraintAmplitudes;

    Float std, var;
    Float mean;

    int nTerms = 5;

    size_t N = 0;

    void empiricalMeanStd() {
        FloatP meanP{0};
        for (size_t i = 0; i < enoki::packets(data->y); ++i) {
            meanP += enoki::packet(data->y, i);
        }
        mean = enoki::hsum(meanP)/N;

        FloatP varP{0};
        for (size_t i = 0; i < enoki::packets(data->y); ++i) {
            FloatP foo = enoki::packet(data->y, i) - mean;
            varP += foo*foo;
        }
        var = enoki::hsum(varP)/(N-1);

        std = std::sqrt(var);
    }

    size_t nModes;
    Float minSigma;
    Float maxSigma;
    const int pdfRes = 1000;
public:


    GaussKeelinMixturePDF(const ProbabilityDistributionSamples& data, size_t nModes, int nTerms) :
        SubspaceState({"A", "mu", "sig", "a", "pdfX", "pdfY", "cdf", "m1", "nNonzeroModes"}, 5, false), constraintAmplitudes(1), nModes(nModes), nTerms(nTerms), data(&data) {

            N = enoki::slices(data.y);
            empiricalMeanStd();

            minSigma = std/50;
            maxSigma = std/3;

            setCoords({std::vector<Float>(nModes + 1, Float(1)/(nModes+1)),
                        std::vector<Float>(nModes, 0),
                         std::vector<Float>(nModes, std/std::max(size_t(5), nModes)),
                         std::vector<Float>{mean*5/6, std, mean/6, -3*std, 0,0,0,0,0,0},
                         std::vector<Float>(pdfRes, 0),
                         std::vector<Float>(pdfRes, 0),
                         std::vector<Float>(pdfRes, 0),
                         std::vector<Float>(1,Float(0)),
                         std::vector<Float>(1,Float(nModes))
                });

            for (size_t i = 0; i < nModes; ++i)
                coords[1][i] = mean-3*std + (i+0.5)*(6*std)/(nModes);

            constraintAmplitudes.link(&coords[0]);

            // for (auto foo : coords) {
            //     for (auto bar : foo)
            //         std::cout << bar << " " ;
            //     std::cout << "\n";
            // }

    }

    std::vector<Float> getInitialConditions() override {
        std::vector<Float> ret{};
        for (size_t i = 0; i < nModes+1; ++i)
            ret.push_back(coords[0][i]);
        for (size_t i = 0; i < nModes; ++i)
            ret.push_back(coords[1][i]);
        for (size_t i = 0; i < nModes; ++i)
            ret.push_back(coords[2][i]);
        for (size_t i = 0; i < 10; ++i)
            ret.push_back(coords[3][i]);

        return ret;
    }

    void setInitialConditions(const std::vector<Float>& ics) override {
        for (size_t i = 0; i < nModes+1; ++i)
            coords[0][i] = ics[i];
        for (size_t i = 0; i < nModes; ++i)
            coords[1][i] = ics[nModes+1+i];
        for (size_t i = 0; i < nModes; ++i)
            coords[2][i] = ics[2*nModes+1+i];
        for (size_t i = 0; i < 10; ++i)
            coords[3][i] = ics[3*nModes+1+i];
    }

    void compute_derived_late(const SharedParams& shared) override {
        static constexpr auto pi = std::acos(-1);

        coords[8][0] = 0;
        for (size_t i = 0; i < nModes+1; ++i)
            if (coords[0][i] > 0.02)
                coords[8][0] += 1;

        Float eps = 1e-5;
        Float dy = (1-2*eps)/(pdfRes-1);
        Float y = eps;

        for (int i = 0; i < pdfRes; ++i) {
            coords[4][i] = keelin_Q(y, coords[3]);
            coords[5][i] = coords[0][nModes]*keelin_pdf_of_y(y, coords[3]);
            coords[6][i] = coords[0][nModes]*y;
            for (int m = 0; m < nModes; ++m) {
                Float arg = coords[4][i] - coords[1][m];
                Float var = coords[2][m]*coords[2][m];
                coords[5][i] += coords[0][m]/(enoki::sqrt(2*pi*var))*enoki::exp(-arg*arg/(2*var));
                coords[6][i] += coords[0][m]*Float(0.5)*(1+enoki::erf(arg/(std::sqrt(Float(2.0))*coords[2][m])));
            }
            y += dy;
        }

    }

    void eval(const SharedParams& shared) override {

        static constexpr auto pi = std::acos(-1);

        FloatP loglikeP(0);

        /* better safe than sorry */
        for (size_t j = nTerms; j < 10; ++j)
            coords[3][j] = 0;

        Float pi2 = pi*pi;
        Float pi4 = pi2*pi2;
        Float pi6 = pi2*pi4;

        Float a1 = coords[3][0];
        Float a2 = coords[3][1];
        Float a3 = coords[3][2];
        Float a4 = coords[3][3];
        Float a5 = coords[3][4];
        Float a6 = coords[3][5];
        Float a7 = coords[3][6];
        Float a8 = coords[3][7];

        empiricalMeanStd();

        for (size_t i = 0; i < enoki::packets(data->y); ++i) {

            FloatP p_i(0);
            FloatP x   = enoki::packet(data->y, i);

            for (size_t m = 0; m < nModes; ++m) {
                /* do not support sample error now */
                // FloatP var = enoki::packet(data->sig, i);
                FloatP var{0};
                FloatP arg = x - coords[1][m];
                // var *= var;
                var += coords[2][m]*coords[2][m];
                p_i += coords[0][m]/(enoki::sqrt(2*pi*var))*enoki::exp(-arg*arg/(2*var));
            }

            std::vector<FloatP> a{coords[3][0], coords[3][1], coords[3][2], coords[3][3], coords[3][4], coords[3][5], coords[3][6], coords[3][7], coords[3][8], coords[3][9]};
            // for (size_t j = 0; j<nTerms; ++j) {
                // a[j] += enoki::packet(data->sig, i)*coords[4][j];
            // }\

            FloatP p_i_keelin = coords[0][nModes]*keelin_pdf(x, a);

            p_i += p_i_keelin;
            p_i[p_i_keelin <= 0] = 1e-80; // unfeasible coefficients "a" manifest like so
            p_i[p_i_keelin > 1e3/std] = 1e-8; // no unfeasible but too weird

            loglikeP += enoki::log(p_i);
        };

        loglike = enoki::hsum(loglikeP);

        /* keelin prior */


        Float m1 = coords[0][nModes]*(a1 + a3/2. + a5/12. + a8/12.);
        for (size_t m = 0; m < nModes; ++m) {
            m1 += coords[0][m]*coords[1][m];
        }

        coords[7][0] = m1;

        /* here we also compute the second central moment including the gaussians, so first of all we use the above m1 to define the center, and recompute the keelin expression for that (note it is not anymore all 2nd order in a) */

        Float m2 = a1*a1 + a1*a3 + a3*a3/3. + a2*a4 + a4*a4/12. + (a1*a5)/6. + (a3*a5)/6. + a5*a5/80. + (2*a2*a6)/3. + (a4*a6)/6. + a6*a6/12. - \
                    2*a1*m1 - a3*m1 - (a5*m1)/6. + m1*m1 + (a2*a2*pi2)/3. + (a3*a3*pi2)/36. + (a2*a6*pi2)/18. + (a6*a6*pi2)/240.;

        /* still need to include the variance caused by the gaussians. Again the center of the moment is m1, but the position of the gaussian is coords[1][m] -
         * a computation shows that the result is the sum of the squared difference of the two and the variance. everything is of course weighted by the cofficents A in coords[0] */

        m2 *= coords[0][nModes];
        for (size_t m = 0; m < nModes; ++m) {
            m2 += coords[0][m]*( (coords[1][m]-m1)*(coords[1][m]-m1) + coords[2][m]*coords[2][m] );
        }


        /* for the rest, we keep the old keelin - only expressions. The goal is to constrain the Keelin part into something nice enough so that it is feasible. The gaussian part can do whatever it wants here, we do not need to  constrain it. We need to include it only in first and second moment, for which we try to fit the empirical ones with the total distribution. */

        Float m3 = (a2*a3*a4)/2. + (a3*a4*a4)/8. + a2*a2*a5 + (a3*a3*a5)/24. + \
(a2*a4*a5)/4. + (a4*a4*a5)/60. + (a3*a5*a5)/120. + a5*a5*a5/3780. + \
(a2*a3*a6)/2. + (a3*a4*a6)/4. + (a2*a5*a6)/3. + (13*a4*a5*a6)/240. + \
(a3*a6*a6)/8. + (3*a5*a6*a6)/80. + a2*a2*a3*pi2 + (a3*a3*a3*pi2)/24. \
+ (a2*a3*a4*pi2)/6. + (a3*a3*a5*pi2)/180. + (5*a2*a3*a6*pi2)/12. + \
(a3*a4*a6*pi2)/40. + (a2*a5*a6*pi2)/90. + (a3*a6*a6*pi2)/24. + \
(a5*a6*a6*pi2)/840.;


    Float m4 = a3*a3*a3*a3/80. + (a2*a3*a3*a4)/2. + 2*a2*a2*a4*a4 + \
(a3*a3*a4*a4)/8. + (a2*a4*a4*a4)/3. + a4*a4*a4*a4/80. + a2*a2*a3*a5 + \
(a3*a3*a3*a5)/24. + (5*a2*a3*a4*a5)/6. + (3*a3*a4*a4*a5)/40. + \
(a2*a2*a5*a5)/6. + (a3*a3*a5*a5)/45. + (a2*a4*a5*a5)/15. + \
(11*a4*a4*a5*a5)/2520. + (a3*a5*a5*a5)/420. + a5*a5*a5*a5/15120. + \
(2*a2*a3*a3*a6)/5. + 3*a2*a2*a4*a6 + (a3*a3*a4*a6)/4. + a2*a4*a4*a6 + \
(23*a4*a4*a4*a6)/360. + (5*a2*a3*a5*a6)/6. + (23*a3*a4*a5*a6)/120. + \
(17*a2*a5*a5*a6)/180. + (a4*a5*a5*a6)/70. + (6*a2*a2*a6*a6)/5. + \
(a3*a3*a6*a6)/8. + a2*a4*a6*a6 + (7*a4*a4*a6*a6)/60. + \
(7*a3*a5*a6*a6)/60. + (67*a5*a5*a6*a6)/6048. + (a2*a6*a6*a6)/3. + \
(11*a4*a6*a6*a6)/120. + (19*a6*a6*a6*a6)/720. + \
(3*a2*a2*a3*a3*pi2)/2. + (a3*a3*a3*a3*pi2)/24. + 2*a2*a2*a2*a4*pi2 + \
(2*a2*a3*a3*a4*pi2)/3. + (a2*a2*a4*a4*pi2)/6. + (a3*a3*a4*a4*pi2)/40. \
+ (a2*a2*a3*a5*pi2)/2. + (a3*a3*a3*a5*pi2)/40. + \
(2*a2*a3*a4*a5*pi2)/45. + (a2*a2*a5*a5*pi2)/90. + \
(11*a3*a3*a5*a5*pi2)/7560. + (8*a2*a2*a2*a6*pi2)/3. + \
(13*a2*a3*a3*a6*pi2)/12. + a2*a2*a4*a6*pi2 + \
(17*a3*a3*a4*a6*pi2)/120. + (a2*a4*a4*a6*pi2)/20. + \
(7*a2*a3*a5*a6*pi2)/36. + (a3*a4*a5*a6*pi2)/105. + \
(11*a2*a5*a5*a6*pi2)/3780. + a2*a2*a6*a6*pi2 + \
(23*a3*a3*a6*a6*pi2)/160. + (23*a2*a4*a6*a6*pi2)/120. + \
(a4*a4*a6*a6*pi2)/224. + (211*a3*a5*a6*a6*pi2)/10080. + \
(a5*a5*a6*a6*pi2)/3360. + (7*a2*a6*a6*a6*pi2)/45. + \
(11*a4*a6*a6*a6*pi2)/840. + (409*a6*a6*a6*a6*pi2)/45360. + \
(7*a2*a2*a2*a2*pi4)/15. + (7*a2*a2*a3*a3*pi4)/30. + \
(7*a3*a3*a3*a3*pi4)/1200. + (7*a2*a2*a2*a6*pi4)/45. + \
(7*a2*a3*a3*a6*pi4)/100. + (7*a2*a2*a6*a6*pi4)/200. + \
(a3*a3*a6*a6*pi4)/160. + (a2*a6*a6*a6*pi4)/240. + \
(7*a6*a6*a6*a6*pi4)/34560.;

    Float m5 = (a2*a3*a3*a3*a4)/4. + (5*a2*a2*a3*a4*a4)/2. + (5*a3*a3*a3*a4*a4)/48. \
+ (5*a2*a3*a4*a4*a4)/6. + (7*a3*a4*a4*a4*a4)/144. + a2*a2*a3*a3*a5 + \
(a3*a3*a3*a3*a5)/48. + 5*a2*a2*a2*a4*a5 + (25*a2*a3*a3*a4*a5)/24. + \
(5*a2*a2*a4*a4*a5)/3. + (7*a3*a3*a4*a4*a5)/48. + \
(13*a2*a4*a4*a4*a5)/72. + (a4*a4*a4*a4*a5)/168. + \
(5*a2*a2*a3*a5*a5)/6. + (7*a3*a3*a3*a5*a5)/288. + \
(11*a2*a3*a4*a5*a5)/36. + (25*a3*a4*a4*a5*a5)/1008. + \
(a2*a2*a5*a5*a5)/18. + (a3*a3*a5*a5*a5)/189. + \
(11*a2*a4*a5*a5*a5)/756. + (a4*a4*a5*a5*a5)/1134. + \
(a3*a5*a5*a5*a5)/2835. + a5*a5*a5*a5*a5/149688. + (a2*a3*a3*a3*a6)/4. \
+ (9*a2*a2*a3*a4*a6)/2. + (5*a3*a3*a3*a4*a6)/24. + \
(5*a2*a3*a4*a4*a6)/2. + (11*a3*a4*a4*a4*a6)/48. + 4*a2*a2*a2*a5*a6 + \
(13*a2*a3*a3*a5*a6)/12. + (15*a2*a2*a4*a5*a6)/4. + \
(11*a3*a3*a4*a5*a6)/32. + (3*a2*a4*a4*a5*a6)/4. + \
(235*a4*a4*a4*a5*a6)/6048. + (59*a2*a3*a5*a5*a6)/144. + \
(227*a3*a4*a5*a5*a6)/3024. + (a2*a5*a5*a5*a6)/42. + \
(149*a4*a5*a5*a5*a6)/45360. + 2*a2*a2*a3*a6*a6 + \
(5*a3*a3*a3*a6*a6)/48. + (5*a2*a3*a4*a6*a6)/2. + \
(19*a3*a4*a4*a6*a6)/48. + 2*a2*a2*a5*a6*a6 + (19*a3*a3*a5*a6*a6)/96. \
+ (23*a2*a4*a5*a6*a6)/24. + (131*a4*a4*a5*a6*a6)/1512. + \
(643*a3*a5*a5*a6*a6)/12096. + (253*a5*a5*a5*a6*a6)/90720. + \
(5*a2*a3*a6*a6*a6)/6. + (43*a3*a4*a6*a6*a6)/144. + \
(7*a2*a5*a6*a6*a6)/18. + (487*a4*a5*a6*a6*a6)/6048. + \
(a3*a6*a6*a6*a6)/12. + (3*a5*a6*a6*a6*a6)/112. + \
(5*a2*a2*a3*a3*a3*pi2)/3. + (5*a3*a3*a3*a3*a3*pi2)/144. + \
(25*a2*a2*a2*a3*a4*pi2)/3. + (5*a2*a3*a3*a3*a4*pi2)/4. + \
(25*a2*a2*a3*a4*a4*pi2)/12. + (7*a3*a3*a3*a4*a4*pi2)/72. + \
(a2*a3*a4*a4*a4*pi2)/12. + (10*a2*a2*a2*a2*a5*pi2)/3. + \
(25*a2*a2*a3*a3*a5*pi2)/12. + (7*a3*a3*a3*a3*a5*pi2)/144. + \
(5*a2*a2*a2*a4*a5*pi2)/6. + (31*a2*a3*a3*a4*a5*pi2)/72. + \
(a2*a2*a4*a4*a5*pi2)/18. + (a3*a3*a4*a4*a5*pi2)/84. + \
(5*a2*a2*a3*a5*a5*pi2)/36. + (25*a3*a3*a3*a5*a5*pi2)/3024. + \
(11*a2*a3*a4*a5*a5*pi2)/756. + (a2*a2*a5*a5*a5*pi2)/1134. + \
(a3*a3*a5*a5*a5*pi2)/3402. + 10*a2*a2*a2*a3*a6*pi2 + \
(125*a2*a3*a3*a3*a6*pi2)/72. + (15*a2*a2*a3*a4*a6*pi2)/2. + \
(13*a3*a3*a3*a4*a6*pi2)/36. + (5*a2*a3*a4*a4*a6*pi2)/6. + \
(5*a3*a4*a4*a4*a6*pi2)/336. + (20*a2*a2*a2*a5*a6*pi2)/9. + \
(71*a2*a3*a3*a5*a6*pi2)/72. + (13*a2*a2*a4*a5*a6*pi2)/24. + \
(187*a3*a3*a4*a5*a6*pi2)/2016. + (a2*a4*a4*a5*a6*pi2)/42. + \
(97*a2*a3*a5*a5*a6*pi2)/1512. + (a3*a4*a5*a5*a6*pi2)/336. + \
(a2*a5*a5*a5*a6*pi2)/1701. + (35*a2*a2*a3*a6*a6*pi2)/6. + \
(85*a3*a3*a3*a6*a6*pi2)/288. + (89*a2*a3*a4*a6*a6*pi2)/48. + \
(39*a3*a4*a4*a6*a6*pi2)/448. + (3*a2*a2*a5*a6*a6*pi2)/4. + \
(1427*a3*a3*a5*a6*a6*pi2)/12096. + (235*a2*a4*a5*a6*a6*pi2)/2016. + \
(5*a4*a4*a5*a6*a6*pi2)/2016. + (433*a3*a5*a5*a6*a6*pi2)/60480. + \
(13*a5*a5*a5*a6*a6*pi2)/199584. + (41*a2*a3*a6*a6*a6*pi2)/36. + \
(1339*a3*a4*a6*a6*a6*pi2)/9072. + (131*a2*a5*a6*a6*a6*pi2)/1134. + \
(49*a4*a5*a6*a6*a6*pi2)/5760. + (11*a3*a6*a6*a6*a6*pi2)/144. + \
(7331*a5*a6*a6*a6*a6*pi2)/1.08864e6 + (14*a2*a2*a2*a2*a3*pi4)/3. + \
(49*a2*a2*a3*a3*a3*pi4)/36. + (49*a3*a3*a3*a3*a3*pi4)/2160. + \
(7*a2*a2*a2*a3*a4*pi4)/9. + (7*a2*a3*a3*a3*a4*pi4)/60. + \
(7*a2*a2*a3*a3*a5*pi4)/90. + (a3*a3*a3*a3*a5*pi4)/360. + \
(7*a2*a2*a2*a3*a6*pi4)/2. + (77*a2*a3*a3*a3*a6*pi4)/135. + \
(7*a2*a2*a3*a4*a6*pi4)/20. + (a3*a3*a3*a4*a6*pi4)/48. + \
(7*a2*a2*a2*a5*a6*pi4)/135. + (a2*a3*a3*a5*a6*pi4)/30. + \
(371*a2*a2*a3*a6*a6*pi4)/360. + (35*a3*a3*a3*a6*a6*pi4)/576. + \
(a2*a3*a4*a6*a6*pi4)/16. + (a2*a2*a5*a6*a6*pi4)/60. + \
(a3*a3*a5*a6*a6*pi4)/288. + (41*a2*a3*a6*a6*a6*pi4)/288. + \
(7*a3*a4*a6*a6*a6*pi4)/1728. + (a2*a5*a6*a6*a6*pi4)/432. + \
(11*a3*a6*a6*a6*a6*pi4)/1440. + (7*a5*a6*a6*a6*a6*pi4)/57024.;


    Float m6 = a3*a3*a3*a3*a3*a3/448. + (3*a2*a3*a3*a3*a3*a4)/16. + \
3*a2*a2*a3*a3*a4*a4 + (5*a3*a3*a3*a3*a4*a4)/64. + 5*a2*a2*a2*a4*a4*a4 \
+ (5*a2*a3*a3*a4*a4*a4)/4. + (5*a2*a2*a4*a4*a4*a4)/4. + \
(19*a3*a3*a4*a4*a4*a4)/192. + (23*a2*a4*a4*a4*a4*a4)/240. + \
a4*a4*a4*a4*a4*a4/448. + (3*a2*a2*a3*a3*a3*a5)/4. + \
(a3*a3*a3*a3*a3*a5)/64. + 9*a2*a2*a2*a3*a4*a5 + \
(9*a2*a3*a3*a3*a4*a5)/8. + (25*a2*a2*a3*a4*a4*a5)/4. + \
(19*a3*a3*a3*a4*a4*a5)/96. + (23*a2*a3*a4*a4*a4*a5)/24. + \
(163*a3*a4*a4*a4*a4*a5)/4032. + 3*a2*a2*a2*a2*a5*a5 + \
(11*a2*a2*a3*a3*a5*a5)/8. + (5*a3*a3*a3*a3*a5*a5)/192. + \
(5*a2*a2*a2*a4*a5*a5)/2. + (37*a2*a3*a3*a4*a5*a5)/48. + \
(17*a2*a2*a4*a4*a5*a5)/24. + (19*a3*a3*a4*a4*a5*a5)/252. + \
(a2*a4*a4*a4*a5*a5)/14. + (a4*a4*a4*a4*a5*a5)/448. + \
(13*a2*a2*a3*a5*a5*a5)/48. + (127*a3*a3*a3*a5*a5*a5)/12096. + \
(25*a2*a3*a4*a5*a5*a5)/252. + (109*a3*a4*a4*a5*a5*a5)/15120. + \
(11*a2*a2*a5*a5*a5*a5)/1008. + (a3*a3*a5*a5*a5*a5)/720. + \
(47*a2*a4*a5*a5*a5*a5)/15120. + (73*a4*a4*a5*a5*a5*a5)/399168. + \
(139*a3*a5*a5*a5*a5*a5)/1.99584e6 + \
(53*a5*a5*a5*a5*a5*a5)/4.6702656e7 + (9*a2*a3*a3*a3*a3*a6)/56. + \
(21*a2*a2*a3*a3*a4*a6)/4. + (5*a3*a3*a3*a3*a4*a6)/32. + \
12*a2*a2*a2*a4*a4*a6 + (15*a2*a3*a3*a4*a4*a6)/4. + \
5*a2*a2*a4*a4*a4*a6 + (43*a3*a3*a4*a4*a4*a6)/96. + \
(7*a2*a4*a4*a4*a4*a6)/12. + (11*a4*a4*a4*a4*a4*a6)/560. + \
8*a2*a2*a2*a3*a5*a6 + (9*a2*a3*a3*a3*a5*a6)/8. + \
(51*a2*a2*a3*a4*a5*a6)/4. + (43*a3*a3*a3*a4*a5*a6)/96. + \
(7*a2*a3*a4*a4*a5*a6)/2. + (1391*a3*a4*a4*a4*a5*a6)/6048. + \
3*a2*a2*a2*a5*a5*a6 + (11*a2*a3*a3*a5*a5*a6)/12. + \
(31*a2*a2*a4*a5*a5*a6)/16. + (13*a3*a3*a4*a5*a5*a6)/63. + \
(335*a2*a4*a4*a5*a5*a6)/1008. + (239*a4*a4*a4*a5*a5*a6)/15120. + \
(97*a2*a3*a5*a5*a5*a6)/672. + (17*a3*a4*a5*a5*a5*a6)/720. + \
(163*a2*a5*a5*a5*a5*a6)/30240. + (353*a4*a5*a5*a5*a5*a6)/498960. + \
(33*a2*a2*a3*a3*a6*a6)/14. + (5*a3*a3*a3*a3*a6*a6)/64. + \
10*a2*a2*a2*a4*a6*a6 + (15*a2*a3*a3*a4*a6*a6)/4. + \
(15*a2*a2*a4*a4*a6*a6)/2. + (3*a3*a3*a4*a4*a6*a6)/4. + \
(11*a2*a4*a4*a4*a6*a6)/8. + (409*a4*a4*a4*a4*a6*a6)/6048. + \
(13*a2*a2*a3*a5*a6*a6)/2. + (a3*a3*a3*a5*a6*a6)/4. + \
(33*a2*a3*a4*a5*a6*a6)/8. + (937*a3*a4*a4*a5*a6*a6)/2016. + \
(5*a2*a2*a5*a5*a6*a6)/4. + (1091*a3*a3*a5*a5*a6*a6)/8064. + \
(163*a2*a4*a5*a5*a6*a6)/336. + (673*a4*a4*a5*a5*a6*a6)/17280. + \
(4423*a3*a5*a5*a5*a6*a6)/241920. + (5101*a5*a5*a5*a5*a6*a6)/7.98336e6 \
+ (20*a2*a2*a2*a6*a6*a6)/7. + (5*a2*a3*a3*a6*a6*a6)/4. + \
5*a2*a2*a4*a6*a6*a6 + (53*a3*a3*a4*a6*a6*a6)/96. + \
(19*a2*a4*a4*a6*a6*a6)/12. + (359*a4*a4*a4*a6*a6*a6)/3024. + \
(19*a2*a3*a5*a6*a6*a6)/12. + (809*a3*a4*a5*a6*a6*a6)/2016. + \
(113*a2*a5*a5*a6*a6*a6)/504. + (9707*a4*a5*a5*a6*a6*a6)/241920. + \
(5*a2*a2*a6*a6*a6*a6)/4. + (29*a3*a3*a6*a6*a6*a6)/192. + \
(43*a2*a4*a6*a6*a6*a6)/48. + (457*a4*a4*a6*a6*a6*a6)/4032. + \
(1525*a3*a5*a6*a6*a6*a6)/12096. + (3569*a5*a5*a6*a6*a6*a6)/241920. + \
(a2*a6*a6*a6*a6*a6)/5. + (85*a4*a6*a6*a6*a6*a6)/1512. + \
(43*a6*a6*a6*a6*a6*a6)/3780. + (25*a2*a2*a3*a3*a3*a3*pi2)/16. + \
(5*a3*a3*a3*a3*a3*a3*pi2)/192. + (35*a2*a2*a2*a3*a3*a4*pi2)/2. + \
(5*a2*a3*a3*a3*a3*a4*pi2)/3. + 10*a2*a2*a2*a2*a4*a4*pi2 + \
(65*a2*a2*a3*a3*a4*a4*pi2)/8. + (19*a3*a3*a3*a3*a4*a4*pi2)/96. + \
(5*a2*a2*a2*a4*a4*a4*pi2)/3. + (17*a2*a3*a3*a4*a4*a4*pi2)/24. + \
(a2*a2*a4*a4*a4*a4*pi2)/16. + (5*a3*a3*a4*a4*a4*a4*pi2)/448. + \
15*a2*a2*a2*a2*a3*a5*pi2 + (35*a2*a2*a3*a3*a3*a5*pi2)/8. + \
(19*a3*a3*a3*a3*a3*a5*pi2)/288. + (65*a2*a2*a2*a3*a4*a5*pi2)/6. + \
(37*a2*a3*a3*a3*a4*a5*pi2)/24. + (35*a2*a2*a3*a4*a4*a5*pi2)/24. + \
(163*a3*a3*a3*a4*a4*a5*pi2)/2016. + (a2*a3*a4*a4*a4*a5*pi2)/21. + \
(5*a2*a2*a2*a2*a5*a5*pi2)/6. + (23*a2*a2*a3*a3*a5*a5*pi2)/24. + \
(19*a3*a3*a3*a3*a5*a5*pi2)/756. + (a2*a2*a2*a4*a5*a5*pi2)/3. + \
(43*a2*a3*a3*a4*a5*a5*pi2)/252. + (11*a2*a2*a4*a4*a5*a5*pi2)/504. + \
(a3*a3*a4*a4*a5*a5*pi2)/224. + (31*a2*a2*a3*a5*a5*a5*pi2)/756. + \
(109*a3*a3*a3*a5*a5*a5*pi2)/45360. + (2*a2*a3*a4*a5*a5*a5*pi2)/567. + \
(a2*a2*a5*a5*a5*a5*pi2)/3024. + (73*a3*a3*a5*a5*a5*a5*pi2)/1.197504e6 \
+ 20*a2*a2*a2*a3*a3*a6*pi2 + (205*a2*a3*a3*a3*a3*a6*pi2)/96. + \
25*a2*a2*a2*a2*a4*a6*pi2 + (95*a2*a2*a3*a3*a4*a6*pi2)/4. + \
(59*a3*a3*a3*a3*a4*a6*pi2)/96. + 10*a2*a2*a2*a4*a4*a6*pi2 + \
(69*a2*a3*a3*a4*a4*a6*pi2)/16. + (23*a2*a2*a4*a4*a4*a6*pi2)/24. + \
(17*a3*a3*a4*a4*a4*a6*pi2)/112. + (5*a2*a4*a4*a4*a4*a6*pi2)/224. + \
(55*a2*a2*a2*a3*a5*a6*pi2)/3. + (385*a2*a3*a3*a3*a5*a6*pi2)/144. + \
(59*a2*a2*a3*a4*a5*a6*pi2)/8. + (2293*a3*a3*a3*a4*a5*a6*pi2)/6048. + \
(211*a2*a3*a4*a4*a5*a6*pi2)/336. + (5*a3*a4*a4*a4*a5*a6*pi2)/504. + \
(17*a2*a2*a2*a5*a5*a6*pi2)/18. + (155*a2*a3*a3*a5*a5*a6*pi2)/336. + \
(3*a2*a2*a4*a5*a5*a6*pi2)/14. + (97*a3*a3*a4*a5*a5*a6*pi2)/2520. + \
(a2*a4*a4*a5*a5*a6*pi2)/112. + (407*a2*a3*a5*a5*a5*a6*pi2)/22680. + \
(13*a3*a4*a5*a5*a5*a6*pi2)/16632. + \
(73*a2*a5*a5*a5*a5*a6*pi2)/598752. + 15*a2*a2*a2*a2*a6*a6*pi2 + \
(65*a2*a2*a3*a3*a6*a6*pi2)/4. + (343*a3*a3*a3*a3*a6*a6*pi2)/768. + \
(50*a2*a2*a2*a4*a6*a6*pi2)/3. + (239*a2*a3*a3*a4*a6*a6*pi2)/32. + \
(7*a2*a2*a4*a4*a6*a6*pi2)/2. + (4303*a3*a3*a4*a4*a6*a6*pi2)/8064. + \
(11*a2*a4*a4*a4*a6*a6*pi2)/56. + (5*a4*a4*a4*a4*a6*a6*pi2)/2304. + \
(22*a2*a2*a3*a5*a6*a6*pi2)/3. + (8947*a3*a3*a3*a5*a6*a6*pi2)/24192. + \
(3487*a2*a3*a4*a5*a6*a6*pi2)/2016. + \
(929*a3*a4*a4*a5*a6*a6*pi2)/13440. + \
(335*a2*a2*a5*a5*a6*a6*pi2)/1008. + \
(6689*a3*a3*a5*a5*a6*a6*pi2)/120960. + \
(239*a2*a4*a5*a5*a6*a6*pi2)/5040. + (85*a4*a4*a5*a5*a6*a6*pi2)/88704. \
+ (4073*a3*a5*a5*a5*a6*a6*pi2)/1.99584e6 + \
(281*a5*a5*a5*a5*a6*a6*pi2)/2.0756736e7 + \
(25*a2*a2*a2*a6*a6*a6*pi2)/3. + (47*a2*a3*a3*a6*a6*a6*pi2)/12. + \
(55*a2*a2*a4*a6*a6*a6*pi2)/12. + (4205*a3*a3*a4*a6*a6*a6*pi2)/6048. + \
(409*a2*a4*a4*a6*a6*a6*pi2)/756. + (563*a4*a4*a4*a6*a6*a6*pi2)/40320. \
+ (91*a2*a3*a5*a6*a6*a6*pi2)/72. + \
(49387*a3*a4*a5*a6*a6*a6*pi2)/362880. + \
(673*a2*a5*a5*a6*a6*a6*pi2)/12960. + \
(197*a4*a5*a5*a6*a6*a6*pi2)/55440. + (95*a2*a2*a6*a6*a6*a6*pi2)/48. + \
(1837*a3*a3*a6*a6*a6*a6*pi2)/6048. + \
(1795*a2*a4*a6*a6*a6*a6*pi2)/3024. + \
(141*a4*a4*a6*a6*a6*a6*pi2)/4480. + \
(58223*a3*a5*a6*a6*a6*a6*pi2)/725760. + \
(146429*a5*a5*a6*a6*a6*a6*pi2)/4.790016e7 + \
(457*a2*a6*a6*a6*a6*a6*pi2)/2016. + \
(21757*a4*a6*a6*a6*a6*a6*pi2)/725760. + \
(3737*a6*a6*a6*a6*a6*a6*pi2)/362880. + (77*a2*a2*a2*a2*a3*a3*pi4)/4. \
+ (91*a2*a2*a3*a3*a3*a3*pi4)/24. + (133*a3*a3*a3*a3*a3*a3*pi4)/2880. \
+ 7*a2*a2*a2*a2*a2*a4*pi4 + (28*a2*a2*a2*a3*a3*a4*pi4)/3. + \
(553*a2*a3*a3*a3*a3*a4*pi4)/720. + (7*a2*a2*a2*a2*a4*a4*pi4)/12. + \
(21*a2*a2*a3*a3*a4*a4*pi4)/40. + (a3*a3*a3*a3*a4*a4*pi4)/64. + \
(35*a2*a2*a2*a2*a3*a5*pi4)/12. + (371*a2*a2*a3*a3*a3*a5*pi4)/360. + \
(163*a3*a3*a3*a3*a3*a5*pi4)/8640. + (14*a2*a2*a2*a3*a4*a5*pi4)/45. + \
(a2*a3*a3*a3*a4*a5*pi4)/15. + (7*a2*a2*a2*a2*a5*a5*pi4)/180. + \
(11*a2*a2*a3*a3*a5*a5*pi4)/360. + (a3*a3*a3*a3*a5*a5*pi4)/960. + \
14*a2*a2*a2*a2*a2*a6*pi4 + (287*a2*a2*a2*a3*a3*a6*pi4)/12. + \
(1519*a2*a3*a3*a3*a3*a6*pi4)/720. + (35*a2*a2*a2*a2*a4*a6*pi4)/6. + \
(679*a2*a2*a3*a3*a4*a6*pi4)/120. + (a3*a3*a3*a3*a4*a6*pi4)/6. + \
(7*a2*a2*a2*a4*a4*a6*pi4)/20. + (3*a2*a3*a3*a4*a4*a6*pi4)/16. + \
(427*a2*a2*a2*a3*a5*a6*pi4)/180. + (959*a2*a3*a3*a3*a5*a6*pi4)/2160. \
+ (a2*a2*a3*a4*a5*a6*pi4)/5. + (a3*a3*a3*a4*a5*a6*pi4)/72. + \
(11*a2*a2*a2*a5*a5*a6*pi4)/540. + (a2*a3*a3*a5*a5*a6*pi4)/80. + \
(35*a2*a2*a2*a2*a6*a6*pi4)/4. + (4333*a2*a2*a3*a3*a6*a6*pi4)/480. + \
(925*a3*a3*a3*a3*a6*a6*pi4)/3456. + (161*a2*a2*a2*a4*a6*a6*pi4)/72. + \
(19*a2*a3*a3*a4*a6*a6*pi4)/16. + (3*a2*a2*a4*a4*a6*a6*pi4)/32. + \
(7*a3*a3*a4*a4*a6*a6*pi4)/384. + (1103*a2*a2*a3*a5*a6*a6*pi4)/1440. + \
(283*a3*a3*a3*a5*a6*a6*pi4)/5760. + (a2*a3*a4*a5*a6*a6*pi4)/24. + \
(a2*a2*a5*a5*a6*a6*pi4)/160. + (17*a3*a3*a5*a5*a6*a6*pi4)/12672. + \
(49*a2*a2*a2*a6*a6*a6*pi4)/18. + (2507*a2*a3*a3*a6*a6*a6*pi4)/1728. + \
(11*a2*a2*a4*a6*a6*a6*pi4)/24. + (493*a3*a3*a4*a6*a6*a6*pi4)/5760. + \
(7*a2*a4*a4*a6*a6*a6*pi4)/576. + (323*a2*a3*a5*a6*a6*a6*pi4)/2880. + \
(7*a3*a4*a5*a6*a6*a6*pi4)/2376. + (17*a2*a5*a5*a6*a6*a6*pi4)/19008. + \
(409*a2*a2*a6*a6*a6*a6*pi4)/864. + (797*a3*a3*a6*a6*a6*a6*pi4)/9216. \
+ (563*a2*a4*a6*a6*a6*a6*pi4)/11520. + \
(7*a4*a4*a6*a6*a6*a6*pi4)/11264. + \
(9517*a3*a5*a6*a6*a6*a6*pi4)/1.52064e6 + \
(581*a5*a5*a6*a6*a6*a6*pi4)/1.1860992e7 + \
(141*a2*a6*a6*a6*a6*a6*pi4)/3200. + \
(1627*a4*a6*a6*a6*a6*a6*pi4)/760320. + \
(13063*a6*a6*a6*a6*a6*a6*pi4)/7.6032e6 + \
(31*a2*a2*a2*a2*a2*a2*pi6)/21. + (155*a2*a2*a2*a2*a3*a3*pi6)/84. + \
(31*a2*a2*a3*a3*a3*a3*pi6)/112. + (31*a3*a3*a3*a3*a3*a3*pi6)/9408. + \
(31*a2*a2*a2*a2*a2*a6*pi6)/42. + (31*a2*a2*a2*a3*a3*a6*pi6)/28. + \
(155*a2*a3*a3*a3*a3*a6*pi6)/1568. + (31*a2*a2*a2*a2*a6*a6*pi6)/112. + \
(465*a2*a2*a3*a3*a6*a6*pi6)/1568. + \
(155*a3*a3*a3*a3*a6*a6*pi6)/16128. + \
(155*a2*a2*a2*a6*a6*a6*pi6)/2352. + (155*a2*a3*a3*a6*a6*a6*pi6)/4032. \
+ (155*a2*a2*a6*a6*a6*a6*pi6)/16128. + \
(155*a3*a3*a6*a6*a6*a6*pi6)/78848. + \
(31*a2*a6*a6*a6*a6*a6*pi6)/39424. + \
(31*a6*a6*a6*a6*a6*a6*pi6)/1.118208e6;


        /* variance of empirical mean */
        Float w = var/N;
        // std::cout << w << "\t";
        Float t1 = 4*Float(0.5)*(m1-mean)*(m1-mean)/w;

        /* variance of empirical variance - estimated as var^2 / N */
        w *= var;
        Float t2 = 4*Float(0.5)*(m2-var)*(m2-var)/w;

        w *= var*(2*2);
        Float t3 = 6*Float(0.5)*m3*m3/w;

        w *= var*(3*3);
        Float k4 = m4 - 3*m2*m2;
        Float t4 = 8*Float(0.5)*k4*k4/w;

        w *= var*(4*4);
        Float k5 = m5 - 10*m3*m2;
        Float t5 = 12*Float(0.5)*k5*k5/w;

        w *= var*(5*5);
        Float k6 = m6 - 15*m4*m2 - 10*m3*m3 + 30*m2*m2*m2;
        Float t6 = 12*Float(0.5)*k6*k6/w;

        loglike -= t1;
        loglike -= t2;
        loglike -= t3;
        loglike -= t4;
        loglike -= t5;
        loglike -= t6;

        /* gaussian prior around for quantities that have even number of terms, therefore for y->1-y they don't do x->-x, unlike the odd terms,
         * so the even terms are zero for all symmetric distributions, and we want something not too skewed. */
        Float t0 = 0;
        for (auto j : std::vector<size_t>{0, 2, 4, 7,8})
            t0 += 0.025*coords[3][j]*coords[3][j]*0.5/(var);

        // loglike -= t0;
        // std::cout << N << " " << loglike << ": mean=" << mean << " [" << m1 <<  " " << m2 << "(" << var << ") " << m3 << " " << m4 << " " << m5 << " " << m6 << "] " << t0 << " ( " << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << " " << t6 << ")\n";


    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = std::dynamic_pointer_cast<GaussKeelinMixturePDF>(copy());

        Float propRatio = 1;

        /* sometimes mix various steps (and sometimes make them larger) to help get unstuck.
         * an easy way (not the fastest, but let's not worry about this) is to loop through multiple times.
         * large steps are not often accepted but should be tried a lot to help convergence. */
        int nSteps = rnd.nextFloat() * 3 * nModes;

        for (int k = 0; k < nSteps; ++k) {
            Float stepkind = rnd.nextFloat();
            Float ampstepthresh = 0.5;
            Float otherthresh = 0.75;

            if (stepkind < ampstepthresh) {
                int from, to;
                Float val;
                newstate->constraintAmplitudes.step(rnd, std::min(2*stepsizeCorrectionFac/nModes/std::sqrt(N), Float(2.0)/nModes), from, to, val, propRatio);
            }
            else {
                //
                if (rnd.nextFloat() < 0.6) {
                //for (size_t i = 0; i < 1; ++i) {

                    size_t whichMode = std::min(size_t(rnd.nextFloat()*nModes), nModes-1);

                    if (stepkind < otherthresh) {

                        if (rnd.nextFloat() < 0.3f)
                            newstate->coords[1][whichMode] += (rnd.nextFloat()-0.5)*8*std*std::min(stepsizeCorrectionFac, Float(1));
                        else
                            newstate->coords[1][whichMode] += 8*(rnd.nextFloat()-0.5)/std::sqrt(N)*std*std::min(stepsizeCorrectionFac, Float(1));


                        bound(newstate->coords[1][whichMode], mean-4*std, mean+4*std);

                    }
                    else {

                        /*sometimes attempt large step: sigma might run away to too large values while amplitude is small, and then amplitude is stuck at 0
                         * if the data follows a narrowly peaked distribution.
                         * it is important to be able to have a shortcut to small sigma so amplitude can raise again in that case */
                        if (rnd.nextFloat() < 0.3f)
                            newstate->coords[2][whichMode] += (rnd.nextFloat()-0.5)*(maxSigma-minSigma)*std::min(stepsizeCorrectionFac, Float(1));
                        else
                            newstate->coords[2][whichMode] += (rnd.nextFloat()-0.5)*(maxSigma-minSigma)/std::sqrt(N)*std::min(stepsizeCorrectionFac, Float(1));

                        bound(newstate->coords[2][whichMode], minSigma, maxSigma);
                    }
                }
                else {
                    int idx = std::min(int(rnd.nextFloat()*nTerms), nTerms-1);
                    // int idx2 = std::min(int(rnd.nextFloat()*2), 1);

                    newstate->coords[3][idx] += (rnd.nextFloat()-0.5)*4*std/std::sqrt(N)*std::min(stepsizeCorrectionFac, Float(1));
                    }
                }
        }

        newstate->eval(shared);
        return Proposal{newstate, propRatio};

    }

    std::shared_ptr<SubspaceState> copy() const override {
        auto foo = new GaussKeelinMixturePDF(*this);
        auto& f = foo->getCoordsAt("A");
        foo->constraintAmplitudes.link(&f);
        //foo->constraint.link(&(foo->getCoords()[0]));
        return std::shared_ptr<SubspaceState>(foo);
    }
private:

    const ProbabilityDistributionSamples* data;

};

