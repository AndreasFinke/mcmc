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

    ProbabilityDistributionSamples(const py::array_t<Float> samples, const py::array_t<Float> sigmas) {
       
        /* round down to nearest multiple of PACKET_SIZE, throw away rest */

        int size = int(samples.size()/PACKET_SIZE)*PACKET_SIZE;

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


#endif
