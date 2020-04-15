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
            discontinuousVals[i]  = dis(1, i);
            if (dis(2, i) >= 0) { // treat as given fixed param
                discontinuousValsFixed[i]  = false;
            }
            else {
                discontinuousValsFixed[i]  = true;
            }
        }
    }

    std::vector<Float> deathsPerDay, deathsSigma;
    std::vector<int> discontinuousDays;
    std::vector<Float> discontinuousVals;
    std::vector<bool> discontinuousValsFixed;
    
    Float initialBetaMild = 2;
    Float initialBetaHigh = 7;
    Float initialDelay = 14;
    Float initialMissedDeaths = 30;

    bool computeR = false;
    bool computeOnlyLikelihood = false;

    int fixBehaviorInAdvance = 14;
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
    Float probICUIfSerious = 0.3;
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
            } else { /* mild (at first)  */

                mild[t2i(t)] += 1;
                infectiousMild[t2i(t)] += 1;

                /* however, one day before (highly) infectious while incubating! */
                if (t2i(t) - 1 > 0) {
                    infectiousHigh[t2i(t) - 1] += 1;
                    infectiousHigh[t2i(t)] -= 1;
                }
                double x = rnd.nextDouble();
                /* remaining mild */
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
    static constexpr size_t nTrajectories = 10000000;

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

    int t2i(double t) const {
        int ret = size_t(nGrid*(t/nDays));
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
    int nPredictDays, maxDelayDaysTilData;
    int popSize;
    int nDaysTotal;

    DiseaseSpread(const DiseaseData& data, const DiseaseParams& params, int popSize, double cap0, double capIncrRate, int maxDelayDaysTilData, size_t nPredictDays) : SubspaceState({"behavior", "discontinuousVals", "betaMild", "betaHigh", "delay", "missedDeaths", "mildlyInfectious", "highlyInfectious", "incubating", "asymptomatic", "mild", "serious", "recovered", "dead", "capacity", "totalBehavior", "R"}, false, 11), data(data), params(params), traj(params), nPredictDays(nPredictDays), maxDelayDaysTilData(maxDelayDaysTilData), popSize(popSize) {
 
        //requestedSharedNames = {};
        nDaysTotal = maxDelayDaysTilData + nPredictDays + data.deathsPerDay.size();

        int Rsize = 1;
        if (data.computeR)
            Rsize = nDaysTotal;

        if (!data.computeOnlyLikelihood) { 
            setCoords( {std::vector(data.deathsPerDay.size()-data.fixBehaviorInAdvance, Float(1)),
                    data.discontinuousVals, 
                    {data.initialBetaMild}, {data.initialBetaHigh}, {data.initialDelay}, {data.initialMissedDeaths}, 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(1)),
                    std::vector(Rsize, Float(0))} );
        }
        else { /*spare some allocations... */
            setCoords( {std::vector(data.deathsPerDay.size()-data.fixBehaviorInAdvance, Float(1)),
                    data.discontinuousVals, 
                    {data.initialBetaMild}, {data.initialBetaHigh}, {data.initialDelay}, {1}, 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(1, Float(0)), 
                    std::vector(nDaysTotal, Float(0)), 
                    std::vector(nDaysTotal, Float(1)),
                    std::vector(Rsize, Float(0))} );
        }

        for (int i = 0; i < nDaysTotal; ++i) 
            getCoordsAt("capacity")[i] = cap0 + std::max(capIncrRate*(i-maxDelayDaysTilData), Float(0.0));

        for (size_t i = 0; i < data.discontinuousValsFixed.size(); ++i) { 
            if (data.discontinuousValsFixed[i]) { 
                fixed.insert(std::make_pair(1, i));
            }
        }
    }

    ~DiseaseSpread() {} 

    std::vector<Float> sampleInitialConditions(pcg32& rnd) override {
        std::vector<Float> ret = {};
        ret.push_back(10*rnd.nextFloat());
        ret.push_back(10*rnd.nextFloat());
        ret.push_back(maxDelayDaysTilData*rnd.nextFloat());

        if (ret[0] > ret[1])
            std::swap(ret[0], ret[1]);

        setInitialConditions(ret);
        return ret;
    }

    void setInitialConditions(const std::vector<Float>& ics) override {
        getCoordsAt("betaMild")[0] = ics[0];
        getCoordsAt("betaHigh")[0] = ics[1];
        getCoordsAt("delay")[0] = ics[2];
    }

    void eval(const SharedParams& shared) override {

        loglike = 0;

        auto piecewise = [&] (int k) -> Float { 
            int found = -1;
            for (size_t m = 0; m < data.discontinuousDays.size()-1; ++m) {
                if ( (data.discontinuousDays[m] <= k - maxDelayDaysTilData) && (k - maxDelayDaysTilData < data.discontinuousDays[m+1]) ) { 
                    found = m;
                    break;
                } 
            }
            Float pcf = 1; /* default: before first discontinuous day */
            /* last loop excluded last element. we might be behind (if not before, or empty vector) */
            if ( (found == -1) && (k - maxDelayDaysTilData >= data.discontinuousDays.back() )) 
                found = data.discontinuousDays.size() - 1;
            if (found != -1)  /* after discontinuous day found */
                pcf = getCoordsAt("discontinuousVals")[found];

            return pcf;
        };

        auto smooth = [&] (int k) -> Float { 

            Float smoothbehavior = 1;
            if (k >= maxDelayDaysTilData) {
                if (k - maxDelayDaysTilData < getCoordsAt("behavior").size())
                    smoothbehavior = getCoordsAt("behavior")[k-maxDelayDaysTilData];
                else 
                    smoothbehavior = getCoordsAt("behavior").back();
            }
            return smoothbehavior;
        };

        getCoordsAt("totalBehavior").assign(nDaysTotal, Float(1));
        for (int i = maxDelayDaysTilData-1; i < nDaysTotal; ++i) {
            /* piecewise actually containts the square roots which are less clustered around 0, improving convergence dramatically */
            getCoordsAt("totalBehavior")[i] = piecewise(i)*piecewise(i)*smooth(i);
        }

        if (data.computeR) 
            getCoordsAt("R").assign(nDaysTotal, Float(0));

        int start = maxDelayDaysTilData - getCoordsAt("delay")[0];
        Float fractionalDelay = maxDelayDaysTilData - getCoordsAt("delay")[0] - start;
        auto shift_weight = [&] (int shift) -> Float { return (1-shift)*(1-fractionalDelay) + shift*fractionalDelay; };

        for (int shift = 1; shift >= 0; --shift) { 

            /* Patient Zero. Treated as highly infectious on first day and like an average case afterwards, including mildly infectious trajectories, 
             * for simplicity. */
            Float newlyInfected = 1;
            /* null old results */
            //getCoordsAt("mildlyInfectious").assign(size, Float(0));
            //getCoordsAt("highlyInfectious").assign(size, Float(0));
            //getCoordsAt("incubating").assign(size, Float(0));
            //getCoordsAt("asymptomatic").assign(size, Float(0));
            //getCoordsAt("mild").assign(size, Float(0));
            //getCoordsAt("serious").assign(size, Float(0));
            //getCoordsAt("recovered").assign(size, Float(0));
            //getCoordsAt("dead").assign(size, Float(0));

            mildlyInfectiousBuf[shift].assign(nDaysTotal, Float(0));
            highlyInfectiousBuf[shift].assign(nDaysTotal, Float(0));
            incubatingBuf[shift].assign(nDaysTotal, Float(0));
            asymptomaticBuf[shift].assign(nDaysTotal, Float(0));
            mildBuf[shift].assign(nDaysTotal, Float(0));
            seriousBuf[shift].assign(nDaysTotal, Float(0));
            recoveredBuf[shift].assign(nDaysTotal, Float(0));
            deadBuf[shift].assign(nDaysTotal, Float(0));

            auto multset = [&](auto& dest, auto& source, double fac) {
                for (size_t i = 0; i < dest.size(); ++i) { 
                    dest[i] = fac*source[i]; 
                }
            };
            auto multadd = [&](auto& dest, auto& source, double fac) {
                for (size_t i = 0; i < dest.size(); ++i) { 
                    dest[i] += fac*source[i]; 
                }
            };
            for (int i = start+shift; i < nDaysTotal; ++i) {

                /* helper function distributing source on dest starting at k with factor n */
                auto project = [&](size_t k, auto& dest, const auto& source, double n) {
                    double cumsum = 0;
                    for (size_t d = 0; d < traj.nDays && (d + k < dest.size()); ++d) { 
                    //std::cout << "dest.size = " << dest.size() << " d = " << d << " k = " << k << "\n";
                        /* getDay only returns the deltas per day (can be negative)
                         * the cumulative sum is the actual number percentage */
                        cumsum += traj.getDay(d, source);
                        if (cumsum > 1.01) std::cout << "ops\n";
                        dest[d+k] += n*cumsum; 
                    }
                };
                auto add = [&](size_t k, auto& dest, double n) {
                    for (size_t d = 0; d < traj.nDays && (d + k < dest.size()); ++d) { 
                        dest[d+k] += n; 
                    }
                };

                /* project newly infected trajectories into future (starting today) */
                //project(i, getCoordsAt("mildlyInfectious"), traj.infectiousMild, newlyInfected);
                //project(i, getCoordsAt("highlyInfectious"), traj.infectiousHigh, newlyInfected);
                //project(i, getCoordsAt("incubating"), traj.incubating, newlyInfected);
                //project(i, getCoordsAt("asymptomatic"), traj.mild, newlyInfected);
                //project(i, getCoordsAt("mild"), traj.mild, newlyInfected);
                //project(i, getCoordsAt("serious"), traj.serious, newlyInfected);
                //project(i, getCoordsAt("recovered"), traj.recovered, newlyInfected);
                //project(i, getCoordsAt("dead"), traj.dead, newlyInfected);

                project(i, mildlyInfectiousBuf[shift], traj.infectiousMild, newlyInfected);
                project(i, highlyInfectiousBuf[shift], traj.infectiousHigh, newlyInfected);
                project(i, incubatingBuf[shift], traj.incubating, newlyInfected);
                project(i, asymptomaticBuf[shift], traj.mild, newlyInfected);
                project(i, mildBuf[shift], traj.mild, newlyInfected);
                project(i, seriousBuf[shift], traj.serious, newlyInfected);
                project(i, recoveredBuf[shift], traj.recovered, newlyInfected);
                project(i, deadBuf[shift], traj.dead, newlyInfected);
                /* correct for overfull hospitals assuming indepence of extra fatalities from their previous trajectory */
                //Float overCapacity = getCoordsAt("serious")[i]*params.probICUIfSerious - getCoordsAt("capacity")[i];
                Float overCapacity = seriousBuf[shift][i]*params.probICUIfSerious - getCoordsAt("capacity")[i];
                Float extraDeaths  = params.probLethalDailyWhenSeriousUntreated * overCapacity;
                /* unfortunately... */
                if (extraDeaths > 0) {
                    //add(i, getCoordsAt("dead"), extraDeaths);
                    add(i, deadBuf[shift], extraDeaths);
                    /* these people are missing in the future. we assume that probLethalDailyWhenSeriousAtHome is so large 
                     * that they died soon after becoming serious - many on the first day, many of the rest on the second. 
                     * approximating, they died today - then we can correct the future given these precomputed trajectories conditioned on becoming
                     * serious today */ 
                    //project(i, getCoordsAt("serious"), traj.seriousFromSerious, -extraDeaths);
                    //project(i, getCoordsAt("recovered"), traj.recoveredFromSerious, -extraDeaths);
                    //project(i, getCoordsAt("dead"), traj.deadFromSerious, -extraDeaths);
                    project(i, seriousBuf[shift], traj.seriousFromSerious, -extraDeaths);
                    project(i, recoveredBuf[shift], traj.recoveredFromSerious, -extraDeaths);
                    project(i, deadBuf[shift], traj.deadFromSerious, -extraDeaths);
                }


                /* compute delta of infected from today to tomorrow */


                //double nSusceptible = popSize - getCoordsAt("incubating")[i] - getCoordsAt("asymptomatic")[i] - getCoordsAt("mild")[i] - getCoordsAt("serious")[i] - getCoordsAt("recovered")[i] - getCoordsAt("dead")[i];
                //newlyInfected = getCoordsAt("totalBehavior")[i]*nSusceptible/popSize*(getCoordsAt("betaMild")[0]*getCoordsAt("mildlyInfectious")[i] + getCoordsAt("betaHigh")[0]*getCoordsAt("highlyInfectious")[i]);
                Float nSusceptible = popSize - incubatingBuf[shift][i] - asymptomaticBuf[shift][i] - mildBuf[shift][i] - seriousBuf[shift][i] - recoveredBuf[shift][i] - deadBuf[shift][i];
                newlyInfected = getCoordsAt("totalBehavior")[i]*nSusceptible/popSize*(getCoordsAt("betaMild")[0]*mildlyInfectiousBuf[shift][i] + getCoordsAt("betaHigh")[0]*highlyInfectiousBuf[shift][i]);

            } // i 
            
            /* compute R at each day - this depends on the future evolution of the susceptible population, so we had to finish the last loop */

            if (data.computeR) {
                for (int i = 0; i < nDaysTotal; ++i) {
                    Float R = 0;
                    Float mildcumsum = 0, highcumsum = 0;
                    for (int j = 0; j < traj.nDays && (j+i) < nDaysTotal; ++j) {
                        Float nSusceptible = popSize - incubatingBuf[shift][i+j] - asymptomaticBuf[shift][i+j] - mildBuf[shift][i+j] - seriousBuf[shift][i+j] - recoveredBuf[shift][i+j] - deadBuf[shift][i+j];
                        mildcumsum += traj.getMildlyInfectious(j);
                        highcumsum += traj.getHighlyInfectious(j);
                        R += getCoordsAt("totalBehavior")[i+j]*nSusceptible/popSize*(getCoordsAt("betaMild")[0]*mildcumsum + getCoordsAt("betaHigh")[0]*highcumsum);
                    }
                    getCoordsAt("R")[i] += shift_weight(shift)*R;
                    //getCoordsAt("R")[i] = R;
                }
            }

            /* copy weighted result to output */
            if (!data.computeOnlyLikelihood) { 
                if (shift == 1) { 
                multset(getCoordsAt("mildlyInfectious"), mildlyInfectiousBuf[shift], shift_weight(shift));
                multset(getCoordsAt("highlyInfectious"), highlyInfectiousBuf[shift], shift_weight(shift));
                multset(getCoordsAt("incubating"), incubatingBuf[shift], shift_weight(shift));
                multset(getCoordsAt("asymptomatic"), asymptomaticBuf[shift], shift_weight(shift));
                multset(getCoordsAt("mild"), mildBuf[shift], shift_weight(shift));
                multset(getCoordsAt("serious"), seriousBuf[shift], shift_weight(shift));
                multset(getCoordsAt("recovered"), recoveredBuf[shift], shift_weight(shift));
                multset(getCoordsAt("dead"), deadBuf[shift], shift_weight(shift));
                } else /*shift == 0, done after it is == 1! */ {
                multadd(getCoordsAt("mildlyInfectious"), mildlyInfectiousBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("highlyInfectious"), highlyInfectiousBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("incubating"), incubatingBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("asymptomatic"), asymptomaticBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("mild"), mildBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("serious"), seriousBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("recovered"), recoveredBuf[shift], shift_weight(shift));
                multadd(getCoordsAt("dead"), deadBuf[shift], shift_weight(shift));
                }
            }


        } // shift

        /* compute and copy weighted loglikelihood to loglike */
        for (size_t i = 0; i < data.deathsPerDay.size(); ++i)  { 
            
            if (std::isnan(getCoordsAt("dead")[i+maxDelayDaysTilData]) ) {
                /* never accept crazy states with exploding number of cases leading to NaN */
                loglike = -1e10;
                break;
            }

            Float delta = getCoordsAt("dead")[i+maxDelayDaysTilData] - getCoordsAt("missedDeaths")[0] - data.deathsPerDay[i];

            loglike +=(-0.5*delta*delta/(data.deathsSigma[i]*data.deathsSigma[i]));
        }

    } 
#define COUT(str) ; // std::cout << str;
    //Proposal step_impl(pcg32& rnd, const SharedParams& shared) const override {
    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        COUT("\n")
        auto newstate = copy();
        //newstate->getCoordsAt("delay")[0] += 0.001;
        //bound(newstate->getCoordsAt("delay")[0], Float(0), Float(maxDelayDaysTilData-1));
        //newstate->eval(shared);
        //return Proposal{newstate, 1};

        bool big1 = rnd.nextFloat() < 0.6f;
        bool big2 = rnd.nextFloat() < 0.6f;
        bool big3 = rnd.nextFloat() < 0.6f;
        bool big4 = rnd.nextFloat() < 0.6f;
        bool big5 = rnd.nextFloat() < 0.6f;
        bool big6 = rnd.nextFloat() < 0.4f;


        /* start time */
        bool changedBeta = false;
        if (rnd.nextFloat() < 0.4f) {

            COUT("del ");
            Float deltaDelay = stepsizeCorrectionFac*6*(rnd.nextDouble()-0.5);//*10*std::min(stepsizeCorrectionFac, Float(1));

            if (!big1)
                deltaDelay /= 10;
            else
                COUT("(L) ")
            COUT(deltaDelay << " ");
            Float oldDelay = newstate->getCoordsAt("delay")[0];
            Float newDelay = oldDelay + deltaDelay;
            bound(newDelay, Float(5), Float(maxDelayDaysTilData));
            newstate->getCoordsAt("delay")[0] = newDelay;
            /* often, propose also a step in beta that is correlated to preserve death number after tPivot days. This should increase acceptance rate. 
             * Note that the i->j proposal probability agrees with j->i (deltaDelay of opposite sign leads to undoing the beta correction since 
             * corrfac -> corrfac^(-1) ) */
            if (rnd.nextFloat() < 0.8f) {
                
                COUT("(with beta) ")
                Float tPivot = 1+10*rnd.nextDouble();
                Float corrfac = (tPivot + oldDelay)/(tPivot + newDelay);
                newstate->getCoordsAt("betaMild")[0] *= corrfac;
                newstate->getCoordsAt("betaHigh")[0] *= corrfac;

                for (size_t i = 0; i < newstate->getCoordsAt("discontinuousVals").size(); ++i) {
                    if (!data.discontinuousValsFixed[i]) { 
                        newstate->getCoordsAt("discontinuousVals")[i] /= corrfac;
                        bound(newstate->getCoordsAt("discontinuousVals")[i], Float(0), Float(1));
                    }
                }

                /* assume beta's are not increasing beyond their bound here, i.e. bound should be large enough to be never reached given the data.
                 * otherwise, bounding them will destroy what was said in the last comment. Lower bound zero is no issue as we multiplied by a positive number.
                 * Anyway bound them for safety... */
                //changedBeta = true;

                bound(newstate->getCoordsAt("betaMild")[0], Float(0), Float(100));
                bound(newstate->getCoordsAt("betaHigh")[0], Float(0), Float(100));

            }
        }
        
        /* missed Deaths */

        if (rnd.nextFloat() < 0.5f) { 
            Float sample = rnd.nextDouble()-Float(0.5);
             //here we propose small numbers with greater probability 
            //newstate->getCoordsAt("missedDeaths")[0] += sample*sample*std::min(stepsizeCorrectionFac, Float(1));
            newstate->getCoordsAt("missedDeaths")[0] += stepsizeCorrectionFac*sample;
            if (big6) 
                newstate->getCoordsAt("missedDeaths")[0] += stepsizeCorrectionFac*(rnd.nextDouble()-Float(0.5))*10;
                //newstate->getCoordsAt("missedDeaths")[0] += (rnd.nextDouble()-Float(0.5))*10*std::min(stepsizeCorrectionFac, Float(1));
            bound(newstate->getCoordsAt("missedDeaths")[0], Float(0), Float(100));
        }


        /* betas */
        if (rnd.nextFloat() < 0.5f) {
            COUT("bet ")
            if (!big2) 
                newstate->getCoordsAt("betaMild")[0] += stepsizeCorrectionFac*(rnd.nextDouble()-Float(0.5))*Float(0.010);
                //newstate->getCoordsAt("betaMild")[0] += (rnd.nextDouble()-Float(0.5))*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
            else { 
                newstate->getCoordsAt("betaMild")[0] += stepsizeCorrectionFac*(rnd.nextDouble()-Float(0.5))*Float(0.1);
                //newstate->getCoordsAt("betaMild")[0] += (rnd.nextDouble()-Float(0.5))*std::min(stepsizeCorrectionFac, Float(1));
                COUT("(L) ")
            }
            if (!big3) 
                newstate->getCoordsAt("betaHigh")[0] += stepsizeCorrectionFac*(rnd.nextDouble()-0.5)*Float(0.010);
                //newstate->getCoordsAt("betaHigh")[0] += (rnd.nextDouble()-0.5)*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
            else { 
                newstate->getCoordsAt("betaHigh")[0] += stepsizeCorrectionFac*(rnd.nextDouble()-0.5)*Float(0.1);
                //newstate->getCoordsAt("betaHigh")[0] += (rnd.nextDouble()-0.5)*std::min(stepsizeCorrectionFac, Float(1));
                COUT("(L) ")
            }

            bound(newstate->getCoordsAt("betaMild")[0], Float(0), Float(10));
            bound(newstate->getCoordsAt("betaHigh")[0], Float(0), Float(10));
            if (newstate->getCoordsAt("betaHigh")[0] < newstate->getCoordsAt("betaMild")[0])
                    std::swap(newstate->getCoordsAt("betaHigh")[0], newstate->getCoordsAt("betaMild")[0]);
        }

        /* discont. vals */

        if (rnd.nextFloat() < 0.6f) { 
            COUT("discon ")

            /* how many to sample? */
            float nNotFixed = 0;
            for (size_t i = 0; i < newstate->getCoordsAt("discontinuousVals").size(); ++i)
                if (!data.discontinuousValsFixed[i])
                    nNotFixed += 1;
            nNotFixed = 0;

            int start = 0;
            int stop = newstate->getCoordsAt("discontinuousVals").size();
            int incr = 1;
            if (rnd.nextDouble() < 0.5) { 
                std::swap(start, stop);
                start--;
                stop--;
                incr = -1;
            }
            if (big4) 
                COUT("(L)(")
            else
                COUT("(")
            for (size_t i = start; i != stop; i += incr) {
                /* jump over fixed ones */
                if (data.discontinuousValsFixed[i])
                    continue;

                /* sample each not fixed with probability 2/nNotFixed, to do usually about two but sometimes more or less - good if they are (anti)correlated, allowing larger steps */
                float x = rnd.nextFloat();
                if (x < 2/nNotFixed) {
                    COUT(i << " ")
                    if (!big4) 
                        newstate->getCoordsAt("discontinuousVals")[i] += stepsizeCorrectionFac*(rnd.nextDouble()-Float(0.5))*Float(0.001);
                        //newstate->getCoordsAt("discontinuousVals")[i] += (rnd.nextDouble()-Float(0.5))*Float(0.1)*std::min(stepsizeCorrectionFac, Float(1));
                    else { 
                        newstate->getCoordsAt("discontinuousVals")[i] += stepsizeCorrectionFac*(rnd.nextDouble()-Float(0.5))*Float(0.01);
                        //newstate->getCoordsAt("discontinuousVals")[i] += (rnd.nextDouble()-Float(0.5))*std::min(stepsizeCorrectionFac, Float(1));
                    }

                    /* bound between left and right Val */
                    Float lower = 0;
                    Float upper = 1;
                    if (i >= 1)
                        upper = newstate->getCoordsAt("discontinuousVals")[i-1];
                    if (i < newstate->getCoordsAt("discontinuousVals").size()-1) 
                        lower = newstate->getCoordsAt("discontinuousVals")[i+1];

                    bound(newstate->getCoordsAt("discontinuousVals")[i], lower, upper);
                }
            }
            COUT(") ")
        }
        /* behaviorfunc */

        if (rnd.nextFloat() < 0.3f) { 
            COUT("beh ")
            Float T = data.deathsPerDay.size(); 
            double x = rnd.nextDouble();
            Float omega = 2*Float(3.1415)/T * 5 * x*x;
            //Float A     = std::min(Float(1), stepsizeCorrectionFac)*0.1*(rnd.nextDouble()-0.5); [>neg A realized by phase <]
            Float A     = stepsizeCorrectionFac*0.0005*(rnd.nextDouble()-0.5); /*neg A realized by phase */
            if (big5)  { 
                //A += std::min(Float(1), stepsizeCorrectionFac)*(rnd.nextDouble()-0.5); [>neg A realized by phase <]
                A += stepsizeCorrectionFac*0.0005*(rnd.nextDouble()-0.5); /*neg A realized by phase */
                COUT("(L) ")
            }

            size_t size = newstate->getCoordsAt("behavior").size();
            //for (size_t i = maxDelayDaysTilData; i < size; ++i) {
                //[> do not predict behavior after 2 weeks before last data point <] 
                //int thresh = size - nPredictDays - 14;
                //if (i < thresh)
                    //newstate->getCoordsAt("behavior")[i] += A*(std::cos(omega*(i-maxDelayDaysTilData))-1);
                //else [>instead fix it <]
                    //newstate->getCoordsAt("behavior")[i] = newstate->getCoordsAt("behavior")[thresh-1];
            //}
            for (size_t i = 0; i < size; ++i) {
                    newstate->getCoordsAt("behavior")[i] += A*(std::cos(omega*i)-1);
                    bound(newstate->getCoordsAt("behavior")[i], Float(0), Float(2));
            }
        }
            
        newstate->eval(shared);
        return Proposal{newstate, 1};

    }

    void force_bounds() override {

        //return;
        bound(getCoordsAt("delay")[0], Float(5), Float(maxDelayDaysTilData));
        bound(getCoordsAt("missedDeaths")[0], Float(0), Float(100));

        Float lastVal = 1;
        for (size_t i = 0; i < getCoordsAt("discontinuousVals").size(); ++i) {
            if (data.discontinuousValsFixed[i])
                continue;
            bound(getCoordsAt("discontinuousVals")[i], Float(0), lastVal);
            lastVal = getCoordsAt("discontinuousVals")[i];
        }
        for (size_t i = 0; i < getCoordsAt("behavior").size(); ++i) {
                bound(getCoordsAt("behavior")[i], Float(0), Float(2));
        }
        bound(getCoordsAt("betaMild")[0], Float(0), Float(100));
        bound(getCoordsAt("betaHigh")[0], Float(0), Float(100));
    }

    HAS_STEP

private:


    std::vector<Float> mildlyInfectiousBuf[2];
    std::vector<Float> highlyInfectiousBuf[2];
    std::vector<Float> incubatingBuf[2];
    std::vector<Float> asymptomaticBuf[2];
    std::vector<Float> mildBuf[2];
    std::vector<Float> seriousBuf[2];
    std::vector<Float> recoveredBuf[2];
    std::vector<Float> deadBuf[2];

};

#endif
