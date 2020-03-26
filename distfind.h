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
using FloatP = enoki::Packet<float, PACKET_SIZE>;
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
public: 

    Float lower, upper, binwidth;
    int nBins;

    PiecewiseConstantPDF(const ProbabilityDistributionSamples& data, Float lower, Float upper, int nBins) :
        SubspaceState({"pdf"}), lower(lower), upper(upper), nBins(nBins), data(data) {

        binwidth = (upper-lower)/nBins;

        setCoords({std::vector<Float>(nBins, Float(1)/(upper-lower))}); 

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

        std::shared_ptr<PiecewiseConstantPDF> newstate = std::dynamic_pointer_cast<PiecewiseConstantPDF>(copy());

        /* if acceptance rate is high because data is not contraining enough individual bins, during adjustment, stepsizeCorrectionFac is growing without
         * bounds. Restrict to just the maximum (max over all pdfs) simultaneous minimal (min over all bins) value which is attained for the initial, flat, pdf. Together with the random float this guarantees acceptance of all steps in the flat-like case*/
        newstate->stepsizeCorrectionFac = std::min(stepsizeCorrectionFac, Float(1)/(upper-lower));

        while (true) { 
            
            int from = std::min(int(rnd.nextFloat()*nBins), nBins-1);
            int to   = std::min(int(rnd.nextFloat()*nBins), nBins-1);
            Float val = newstate->stepsizeCorrectionFac*rnd.nextFloat();

            if ( newstate->moveMass(from, to, val)) { //move allowed

                Float newVol = newstate->accessibleStateVol(newstate->stepsizeCorrectionFac);
                Float oldVol = newVol - newstate->accessibleStateVolCorrection(newstate->stepsizeCorrectionFac, from, to, val);

                /* the trans. probs q_ij from state i to state q are not symmetric, but obey
                 * q_ij  vol(reached from i) = q_ji  vol(reached from j) 
                 * because sum_j q_ij = 1 is forcing q_ij to be inverse to the volume reached from i (i.e. that j is summed over) 
                 *
                 * therefore, the factor q_ji/q_ij for the acceptance prob for going i->j is vol(from i)/vol(from j) = oldVol/newVol */

                Float propRatio = oldVol/newVol;
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

                    FloatP sigi = enoki::packet(data.sig, i) + 0.000001;
                    FloatP invsig = 1/sigi;
                    FloatP f = invsig * Float(0.70710678118);
                    FloatP dto = binpos(to) - enoki::packet(data.y,i);
                    FloatP dfrom = binpos(from) - enoki::packet(data.y,i); 
                    Float hw = binwidth*Float(0.5);
                    enoki::packet(newstate->pconv,i) = enoki::packet(pconv,i) + 0.5f*val*(enoki::erf((dto+hw)*f) - enoki::erf((dto-hw)*f) - enoki::erf((dfrom+hw)*f) + enoki::erf((dfrom-hw)*f));

                    loglike_prop_packet += enoki::log(enoki::packet(newstate->pconv,i));
                }

                newstate->loglike = enoki::hsum(loglike_prop_packet);

                return Proposal{newstate, propRatio};
            }
        }

    }

    std::shared_ptr<SubspaceState> copy() const override {
        return std::shared_ptr<SubspaceState>(new PiecewiseConstantPDF(*this));
    }

private:

    const ProbabilityDistributionSamples& data;

    FloatX pconv;
    
    Float binpos(int i) const {
        return lower + (i+Float(0.5))*binwidth;
    }

    bool moveMass(int from, int to, Float val) { 

        if (from == to) return false;

        if (coords[0][from] < val) return false;

        coords[0][to] += val;
        coords[0][from] -= val;

        return true;
    }

    Float accessibleStateVol(Float propMassDistWidth) const { 
        Float ret = 0;
        for (auto f : coords[0]) 
            ret += std::min(f, propMassDistWidth);
        return ret;
    }

    /* returns the difference of the above function and the value it would have yielded before calling moveMass with the params given here */
    Float accessibleStateVolCorrection(Float propMassDistWidth, int from, int to, Float val) const {
        return std::min(propMassDistWidth, coords[0][from]) + std::min(propMassDistWidth, coords[0][to]) - std::min(propMassDistWidth, coords[0][from]+val) - std::min(propMassDistWidth, coords[0][to]-val);
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



#endif
