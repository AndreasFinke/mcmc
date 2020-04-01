/*
    Copyright (c) 2020 Andreas Finke <andreas.finke@unige.ch>

    All rights reserved. Use of this source code is governed by a modified BSD
    license that can be found in the LICENSE file.
*/


#pragma once

#include "mcmc.h"
#include <memory>

#if PY == 1

class DiseaseData {
public:

    DiseaseData(const py::array_t<Float> deathsPerDayAndSigma, const py::array_t<Float> discontinuousDaysAndVals) {

        auto deaths = deathsPerDayAndSigma.unchecked<2>();
        deathsPerDay.resize(deaths.shape(1));
        deathsSigma.resize(deaths.shape(1));

        for (size_t i = 0; i < deathsPerDay.size(); ++i) {
            deathsPerDay[i] = deaths(0, i);
            deathsSigma[i]  = deaths(1, i);
        }

        auto dis = discontinuousDaysAndVals.unchecked<2>();
        discontinuousDays.resize(dis.shape(1));
        discontinuousVals.resize(dis.shape(1));
        discontinuousValsFixed.resize(dis.shape(1));

        for (size_t i = 0; i < discontinuousDays.size(); ++i) {
            discontinuousDays[i] = std::round(dis(0, i));
            if (dis(1, i) >= 0) { // treat as given fixed param
                discontinuousValsFixed[i]  = true;
                discontinuousVals[i]  = dis(1, i);
            }
            else {
                discontinuousValsFixed[i]  = false;
                discontinuousVals[i]  = 0;
            }
        }
    }

    std::vector<Float> deathsPerDay, deathsSigma;
    std::vector<size_t> discontinuousDays;
    std::vector<Float> discontinuousVals;
    std::vector<bool> discontinuousValsFixed;
};

class DiseaseParams {
public:
    /* we have: susecptible, infected in incubation, infected (post incubation), recovered(and immune), dead
     * incubation period is lognormal with (timeIncub, timeIncubSigma)
     * infected post incubation includes the following possibbilities: 
     *     at first, either asymptomatic (with probAsymp)
     *               or  mildly infected (otherwise) , both will recover after +timeMildDuration
     *               unless from the latter they get worse: getting serious (i.e. hospitalized), namely at +timeMildToSerious
     *               with probSerious/(1-probAsymp) (conditioned on not asympt) serious
     *               with outcome recovered at some sampled time timeSeriousToRec
     *               and probLethal/(1-probAsymp) others will be dead at sampled time +timeSeriousToDead
     *
     *    infected (mild) go to infected (serious) with probSerious after timeSeriousFromSymptom
     *    infected (serious) go to dead with probLethalIfSerious after timeLethalAfterSerious 
     */

    Float timeIncub = 4;
    Float timeIncubSigma = 2;
    Float probAsymp  = 0.1;
    Float probSerious = 0.08; /* but not lethal */
    Float probLethal = 0.004;
    Float probLethalDailyWhenSeriousUntreated = 0.2;
    /* probMild = 1 - probAsymp - probSerious - probLethal */
    Float timeMildDuration = 10;
    Float timeMildDurationSigma = 5;
    Float timeMildToSerious = 8;
    Float timeMildToSeriousSigma = 3;
    Float timeSeriousToRec = 10;
    Float timeSeriousToRecSigma = 5;
    Float timeSeriousToDeath = 9;
    Float timeSeriousToDeathSigma = 5;

};

/* cumulative distributions of outcomes of a single infected person after n days */
class AvgDiseaseTrajectory {

public:

    AvgDiseaseTrajectory(const DiseaseParams& p) : incubating{}, asymptomatic{}, mild{}, infectiousMild{}, infectiousHigh{}, serious{}, dead{}, recovered{} {

        pcg32 rnd;
        for (size_t k = 0; k < nTrajectories; ++k) {

            /* we compute the derivative first (quicker, only need to access arrays at to points) and integrate in the end */
            incubating[0] += 1;

            /* sample end of incubation */
            double t = sampleTime(rnd, p.timeIncub, p.timeIncubSigma);
            incubating[t2i(t)] -= 1;

            /* asympyomatic */
            if (rnd.nextDouble() < p.probAsymp) {
                asymptomatic[t2i(t)] += 1;
                infectiousHigh[t2i(t)] += 1;
                /* sample recovery */
                double t2 = sampleTime(rnd, p.timeMildDuration, p.timeMildDurationSigma);
                asymptomatic[t2i(t+t2)] -= 1;
                recovered[t2i(t+t2)] += 1;
                infectiousHigh[t2i(t+t2)] -= 1; // for now 
            } else { /* mild */

                mild[t2i(t)] += 1;
                infectiousMild[t2i(t)] += 1;
                /* remaining mild */
                double x = rnd.nextDouble();
                if (x < 1 - (p.probSerious + p.probLethal)/(1 - p.probAsymp)) {
                    /* sample recovery */
                    double t2 = sampleTime(rnd, p.timeMildDuration, p.timeMildDurationSigma);
                    mild[t2i(t+t2)] -= 1; 
                    infectiousMild[t2i(t+t2)] -= 1;
                    recovered[t2i(t+t2)] += 1;
                }
                else { //serious
                    /* sample start serious */
                    double t2 = sampleTime(rnd, p.timeMildToSerious, p.timeMildToSerious);
                    infectiousMild[t2i(t+t2)] -= 1; /* person isolated in hospital */
                    serious[t2i(t+t2)] += 1;
                    mild[t2i(t+t2)] -= 1;
                    double t3;
                    if (x < 1 - p.probLethal/(1-p.probAsymp)) {
                        t3 = sampleTime(rnd, p.timeSeriousToRec, p.timeSeriousToRecSigma);
                        recovered[t2i(t+t2+t3)] += 1;
                    } 
                    else { 
                        t3 = sampleTime(rnd, p.timeSeriousToDeath, p.timeSeriousToDeathSigma);
                        dead[t2i(t+t2+t3)] += 1;
                    }
                    serious[t2i(t+t2+t3)] -= 1;
                }
            }

            /* we will also need the trajectories conditioned on starting as serious. They have two outcomes,
             * lethal with probLelthal/(probLethal+probSerious) and recovery with prob probSerious(probLethal+probSerious) 
             * (indeed, above prob cond on not asympt. for serious outcome is (probLethal+probSerious)/(1-probAsymp), and we divide 
             * the individual probLethal/(1-probAsymp) and probSerious/(1-probAsymp) by this.) */
            
            double t3;
            double x = rnd.nextDouble(); 
            seriousFromSerious[0] += 1;
            if (x < p.probSerious/(p.probLethal + p.probSerious)) {
                t3 = sampleTime(rnd, p.timeSeriousToRec, p.timeSeriousToRecSigma);
                recoveredFromSerious[t2i(t3)] += 1;
            } 
            else { 
                t3 = sampleTime(rnd, p.timeSeriousToDeath, p.timeSeriousToDeathSigma);
                deadFromSerious[t2i(t3)] += 1;
            }
            seriousFromSerious[t2i(t3)] -= 1;
        }

        incubating[0] /= nTrajectories;
        asymptomatic[0] /= nTrajectories;
        mild[0] /= nTrajectories;
        infectiousMild[0] /= nTrajectories;
        infectiousHigh[0] /= nTrajectories;
        serious[0] /= nTrajectories;
        dead[0] /= nTrajectories;
        recovered[0] /= nTrajectories;
        seriousFromSerious[0] /= nTrajectories;
        recoveredFromSerious[0] /= nTrajectories;
        deadFromSerious[0] /= nTrajectories;
        for (size_t t = 1; t < nGrid; ++t) {
            auto cumsum = [&](auto& a, size_t i) { a[i] = a[i-1] + a[i]/nTrajectories; };
            cumsum(incubating, t);
            cumsum(asymptomatic, t);
            cumsum(mild, t);
            cumsum(infectiousMild, t);
            cumsum(infectiousHigh, t);
            cumsum(serious, t);
            cumsum(dead, t);
            cumsum(recovered, t);
            cumsum(seriousFromSerious, t);
            cumsum(recoveredFromSerious, t);
            cumsum(deadFromSerious, t);
        }
    }

    static constexpr size_t nGrid = 100000;
    static constexpr size_t nDays = 500;
    static constexpr size_t nTrajectories = 40000000;

    std::array<double, nGrid> incubating;
    std::array<double, nGrid> asymptomatic;
    std::array<double, nGrid> mild;
    std::array<double, nGrid> infectiousMild;
    std::array<double, nGrid> infectiousHigh;
    std::array<double, nGrid> serious;
    std::array<double, nGrid> dead;
    std::array<double, nGrid> recovered;
    std::array<double, nGrid> seriousFromSerious;
    std::array<double, nGrid> recoveredFromSerious;
    std::array<double, nGrid> deadFromSerious;

    double getDay(size_t day, const std::array<double, nGrid>& ar) const {

        if (day == 0) 
            return ar[0];
        if (day == nDays)
            return 0;

        size_t i1 = size_t(nGrid*(double(day-1)/nDays));
        size_t i2 = size_t(nGrid*(double(day)/nDays));
        if (i2 > nGrid-1) {
            std::cout << "Access past last day of trajectory!\n";
            return 0;
        }
        return ar[i2]-ar[i1];
    }
    double getIncubating(size_t day) const {
        return getDay(day, incubating); 
    }
    double getAsymptomatic(size_t day) const {
        return getDay(day, asymptomatic); 
    }
    double getMild(size_t day) const {
        return getDay(day, mild); 
    }
    double getHighlyInfectious(size_t day) const {
        return getDay(day, infectiousHigh); 
    }
    double getMildlyInfectious(size_t day) const {
        return getDay(day, infectiousMild); 
    }
    double getSerious(size_t day) const {
        return getDay(day, serious); 
    }
    double getDead(size_t day) const {
        return getDay(day, dead); 
    }
    double getRecovered(size_t day) const {
        return getDay(day, recovered); 
    }
private:

    double sampleTime(pcg32 rnd, double mean, double sig) const {
        double s = std::sqrt(std::log(sig*sig/(mean*mean) + 1));
        double expmu = mean*std::exp(-double(0.5)*s*s);
        double sample;
        do {
            sample = rnd.nextDouble();
        } while (sample <= 0 || sample >= 1); 
        Float n = toNormal(sample)*s;
        return expmu*std::exp(n);
    }

    size_t t2i(double t) const {
        size_t ret = size_t(nGrid*(t/nDays));
        if (ret > nGrid-1) {
            std::cout << "Increase nDays! (t=" << t << ")\n";
            return nGrid-1;
        }
        return ret;
    }

};

class DiseaseSpread : public  SubspaceState {

public:

    const DiseaseData& data;
    const DiseaseParams& params;
    AvgDiseaseTrajectory traj; 
    size_t nPredictDays, maxDelayDaysTilData;
    int popSize;

    DiseaseSpread(const DiseaseData& data, const DiseaseParams& params, int popSize, double cap0, double capIncrRate, int maxDelayDaysTilData, size_t nPredictDays) : SubspaceState({"behavior", "discontinuousVals", "betaMild", "betaHigh", "delay", "mildlyInfectious", "highlyInfectious", "incubating", "asymptomatic", "mild", "serious", "recovered", "dead", "capacity"}, 9, false), data(data), params(params), traj(params), nPredictDays(nPredictDays), maxDelayDaysTilData(maxDelayDaysTilData), popSize(popSize) {
  
        //requestedSharedNames = {};
        size_t nDaysTotal = maxDelayDaysTilData + nPredictDays + data.deathsPerDay.size();

        setCoords( {std::vector(nDaysTotal, Float(1)),
                data.discontinuousVals, 
                {Float(0.1)}, {Float(0.1)}, {Float(10.0)},
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0)), 
                std::vector(nDaysTotal, Float(0))} );

        for (int i = 0; i < nDaysTotal; ++i) 
            getCoordsAt("capacity")[i] = cap0 + std::max(capIncrRate*(i-maxDelayDaysTilData), Float(0.0));
    }

    ~DiseaseSpread() {} 

    void eval(const SharedParams& shared) override {

        loglike = 0;

        size_t size = getCoordsAt("dead").size();
        /* null old results */
        getCoordsAt("mildlyInfectious").assign(size, Float(0));
        getCoordsAt("highlyInfectious").assign(size, Float(0));
        getCoordsAt("incubating").assign(size, Float(0));
        getCoordsAt("asymptomatic").assign(size, Float(0));
        getCoordsAt("mild").assign(size, Float(0));
        getCoordsAt("serious").assign(size, Float(0));
        getCoordsAt("recovered").assign(size, Float(0));
        getCoordsAt("dead").assign(size, Float(0));

        /* Patient Zero. Treated as highly infectious on first day and like an average case afterwards, including mildly infectious trajectories, 
         * for simplicity. */
        double newlyInfected = 1;
        size_t start = maxDelayDaysTilData - getCoordsAt("delay")[0];

        for (size_t i = start; i < size; ++i) {

            /* helper function distributing source on dest starting at k with factor n */
            auto project = [&](size_t k, auto& dest, const auto& source, double n) {
                double cumsum = 0;
                for (size_t d = 0; d < traj.nDays && (d + k < dest.size()); ++d) { 
                //std::cout << "dest.size = " << dest.size() << " d = " << d << " k = " << k << "\n";
                    /* getDay only returns the deltas per day (can be negative)
                     * the cumulative sum is the actual number percentage */
                    cumsum += traj.getDay(d, source);
                    if (cumsum > 1) std::cout << "ops\n";
                    dest[d+k] += n*cumsum; 
                }
            };
            auto add = [&](size_t k, auto& dest, double n) {
                for (size_t d = 0; d < traj.nDays && (d + k < dest.size()); ++d) { 
                    dest[d+k] += n; 
                }
            };

            /* project newly infected trajectories into future (starting today) */
            project(i, getCoordsAt("mildlyInfectious"), traj.infectiousMild, newlyInfected);
            project(i, getCoordsAt("highlyInfectious"), traj.infectiousHigh, newlyInfected);
            project(i, getCoordsAt("incubating"), traj.incubating, newlyInfected);
            project(i, getCoordsAt("asymptomatic"), traj.mild, newlyInfected);
            project(i, getCoordsAt("mild"), traj.mild, newlyInfected);
            project(i, getCoordsAt("serious"), traj.serious, newlyInfected);
            project(i, getCoordsAt("recovered"), traj.recovered, newlyInfected);
            project(i, getCoordsAt("dead"), traj.dead, newlyInfected);

            /* correct for overfull hospitals assuming indepence of extra fatalities from their previous trajectory */
            Float overCapacity = getCoordsAt("serious")[i] - getCoordsAt("capacity")[i];
            Float extraDeaths  = params.probLethalDailyWhenSeriousUntreated * overCapacity;
            /* unfortunately... */
            if (extraDeaths > 0) {
                add(i, getCoordsAt("dead"), extraDeaths);
                /* these people are missing in the future. we assume that probLethalDailyWhenSeriousAtHome is so large 
                 * that they died soon after becoming serious - many on the first day, many of the rest on the second. 
                 * approximating, they died today - then we can correct the future given these precomputed trajectories conditioned on becoming
                 * serious today */ 
                project(i, getCoordsAt("serious"), traj.seriousFromSerious, -extraDeaths);
                project(i, getCoordsAt("recovered"), traj.recoveredFromSerious, -extraDeaths);
                project(i, getCoordsAt("dead"), traj.deadFromSerious, -extraDeaths);
            }

            /* find position of day i in piecewise constant function */
            long found = -1;
            for (size_t m = 0; m < data.discontinuousDays.size()-1; ++m) {
                if ( (data.discontinuousDays[m] <= i - maxDelayDaysTilData) && (i - maxDelayDaysTilData < data.discontinuousDays[m+1]) ) { 
                    found = m;
                    break;
                } 
            }
            /* last loop excluded last element. we might be behind (if not before, or empty vector) */
            if ( (found == -1) && (i - maxDelayDaysTilData >= data.discontinuousDays.back() )) 
                found = data.discontinuousDays.size() - 1;
            Float pcf = 1; /* default: before first discontinuous day */
            if (found != -1)  /* after discontinuous day found */
                pcf = getCoordsAt("discontinuousVals")[found];

            /* compute delta of infected from today to tomorrow */
            double nSusceptible = popSize - getCoordsAt("incubating")[i] - getCoordsAt("asymptomatic")[i] - getCoordsAt("mild")[i] - getCoordsAt("serious")[i] - getCoordsAt("recovered")[i] - getCoordsAt("dead")[i];
            newlyInfected = pcf*getCoordsAt("behavior")[i]*nSusceptible/popSize*(getCoordsAt("betaMild")[0]*getCoordsAt("mildlyInfectious")[i] + getCoordsAt("betaHigh")[0]*getCoordsAt("highlyInfectious")[i]);

        }

        /* compute loglikelihood */
        for (size_t i = 0; i < data.deathsPerDay.size(); ++i)  { 
            Float delta = getCoordsAt("dead")[i+maxDelayDaysTilData] - data.deathsPerDay[i];

            loglike += -0.5*delta*delta/data.deathsSigma[i]*data.deathsSigma[i];
        }
            


    }

    //Proposal step_impl(pcg32& rnd, const SharedParams& shared) const override {
    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = copy();

        /* start time */

        newstate->getCoordsAt("delay")[0] += (rnd.nextDouble()-Float(0.5))*3;
        bound(newstate->getCoordsAt("delay")[0], Float(0.0), Float(maxDelayDaysTilData));

        /* betas */
        newstate->getCoordsAt("betaMild")[0] += (rnd.nextDouble()-Float(0.5))*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
        bound(newstate->getCoordsAt("betaMild")[0], 0.0, 1000.0);
        newstate->getCoordsAt("betaHigh")[0] += (rnd.nextDouble()-0.5)*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
        bound(newstate->getCoordsAt("betaHigh")[0], Float(0.0), Float(1000.0));
        if (newstate->getCoordsAt("betaHigh")[0] < newstate->getCoordsAt("betaMild")[0])
                std::swap(newstate->getCoordsAt("betaHigh")[0], newstate->getCoordsAt("betaMild")[0]);

        /* discont. vals */

        for (size_t i = 0; i < newstate->getCoordsAt("discontinuousVals").size(); ++i) {
            if (data.discontinuousValsFixed[i])
                continue;
            newstate->getCoordsAt("discontinuousVals")[i] += (rnd.nextDouble()-Float(0.5))*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
            bound(newstate->getCoordsAt("discontinuousVals")[i], Float(0.0), Float(1.0));

        }

        /* behaviorfunc */

        Float T = data.deathsPerDay.size(); 
        Float omega = 2*Float(3.1415)/T * 5 * rnd.nextDouble();
        Float A     = std::min(Float(1), stepsizeCorrectionFac)*0.1*(rnd.nextDouble()-0.5); /*neg A realized by phase */

        for (size_t i = maxDelayDaysTilData; i < newstate->getCoordsAt("behavior").size(); ++i) {
            newstate->getCoordsAt("behavior")[i] += A*std::cos(omega*(i-maxDelayDaysTilData));
        }
            
        newstate->eval(shared);
        return Proposal{newstate, 1};

    }

    HAS_STEP

private:

    std::vector<std::vector<Float>> datapoints;

};

#endif
