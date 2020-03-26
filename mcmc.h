/*
    Copyright (c) 2020 Andreas Finke <andreas.finke@unige.ch>

    All rights reserved. Use of this source code is governed by a modified BSD
    license that can be found in the LICENSE file.
*/

#pragma once

bool verbose = false;

#include <iostream> 
#include <vector>
#include <memory>
#include <map>
#include <unordered_set>
#include <list>

#include "timer.h"
#include "pcg32.h"

#if PY == 1

#include <pybind11/numpy.h>

namespace py = pybind11;

#endif

using Float = float;

using SharedParams = std::map<std::string, std::vector<Float>>;

class State; 

class SubspaceState {

friend class State;

public:

    float loglike = 0;
    Float stepsizeCorrectionFac = 1;
    bool derivedOnShared;

    SubspaceState(std::vector<std::string>&& coordNames, bool derivedParamsDependOnSharedParams = false, size_t nDerived = 0) :  derivedOnShared(derivedParamsDependOnSharedParams), nDerived(nDerived) {  
        for (size_t i = 0; i < coordNames.size(); ++i) {
            names[coordNames[i]] = i;
        }
    } 
    //SubspaceState(const SubspaceState& ss) : names(ss.names), coords(ss.coords), nDerived(ss.nDerived), stepsizeCorrectionFac(ss.stepsizeCorrectionFac) {
    //} 

    /* derived class constructor must call this! */
    void setCoords(std::vector<std::vector<Float>> init) {
        coords = init;
    }


    /* the second number is the pij/pji correction factor for the MCMC */
    using Proposal = std::pair<std::shared_ptr<SubspaceState>, Float>;

    /* evaluate likelihood, update loglike and derived parameters, update shared parameters that are owned by this likelihood
     * to the values that were modified by stepping (the coords) or that have otherwise been evaluated (last nDerived vectors of coords array) */
    //Float eval(SharedParams& shared) {

        //eval_likelihood(shared);

        //for (auto&& s : shared) {

            //auto c = getCoords(s.first);
            //if (c) 
                //s.second = *c;

        //}
            
        //return loglike;
    //}

    /* derived class needs to implement these */
    virtual std::shared_ptr<SubspaceState> copy() const = 0;
    virtual void eval(const SharedParams& shared) = 0;
    //virtual Proposal step_impl(pcg32& rnd, const SharedParams& shared) const = 0;
    virtual Proposal step(pcg32& rnd, const SharedParams& shared) const = 0;
    virtual ~SubspaceState() {};

    inline auto& getCoords()  { return coords; }

    inline std::optional<std::vector<Float>> getCoords(const std::string name) const {
        auto it = names.find(name);
        if (it != names.end()) {
            return coords[it->second]; 
        }
        else
            return {};
    }

    inline bool isDerived(const std::string name) const {
        auto it = names.find(name);
        if (it != names.end()) {
            return it->second >= names.size()-nDerived; 
        }
        else
            return {};
    }
    auto getNames() const {
        return names;
    }


    //inline Proposal step(pcg32& rnd, const SharedParams& shared) const {

        //auto ret  = step_impl(rnd, shared);
        //ret.first->eval_likelihood(shared);

        //return ret;
    //}



protected:

    std::vector< std::vector<Float> > coords = {};

    /* maps parameter names to vector index in coord array */
    std::map<std::string, size_t> names = {};

    /* list of names required from other SubspaceStates. Set initially, State::init will access */
    std::vector<std::string> requestedSharedNames;

    /* list of names offered to other SubspaceStates. State::init will build */
    std::unordered_set<std::string> offeredSharedNames;

    /* last nDerived coords are actually derived parameters TODO: need?*/
    size_t nDerived = 0;

    inline std::optional<std::vector<Float>> getSharedCoords(const SharedParams& shared, const std::string name) const {
        auto it = shared.find(name);
        if (it != shared.end()) {
            return it->second; 
        }
        else
            return {};
    }

};

class SmoothnessPrior : public SubspaceState {

public:
    SmoothnessPrior(const std::string& functionName, Float smoothnessScale, Float x1, Float x2) : SubspaceState({}), lower(x1), upper(x2), Lsmooth(smoothnessScale) {

        requestedSharedNames.push_back(functionName);

        /* no coords to set */
    }

    Float lower, upper, Lsmooth; 
    
    void eval(const SharedParams& shared) override {

        auto f = getSharedCoords(shared, requestedSharedNames[0]);
        
        if (!f) {
            std::cout << "SmoothnessPrior cannot evaluate: " << requestedSharedNames[0] << " is not part of the shared parameters.\n";
            return;
        }
        Float dx = (upper-lower)/f->size();  
        loglike = 0;
        for (size_t i = 1; i < f->size()-1; ++i) { 
            Float dd = (*f)[i+1] + (*f)[i-1] - 2*(*f)[i]; 
            loglike -= dd*dd/dx/dx/dx; /*one dx cancels from the integration*/

        }

        loglike *= Lsmooth*Lsmooth/(upper-lower);
    }

    //Proposal step_impl(pcg32& rnd, const SharedParams& shared) const override {
    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = copy();
        return Proposal{newstate, 1};

    }

    std::shared_ptr<SubspaceState> copy() const override {
        return std::shared_ptr<SubspaceState>(new SmoothnessPrior(*this));
    }

};

struct ProposalRecord {
    //size_t which;
    //std::shared_ptr<SubspaceState> prop;
    std::map<size_t, std::shared_ptr<SubspaceState>> prop;
    size_t whichHead;
    SharedParams shared_prop;
    Float deltaloglike;
    Float propasymratio;
    Float weight = 1;
};

class State {


protected:

    SharedParams shared;

public:

    std::vector< std::shared_ptr<SubspaceState> > state;

    /* for each, i-th, SubspaceState in state, we keep a corresponding i-th element in dependencies:
     * a list of the other SubspaceStates that depend on the i-th SubspaceStates, together with a flag if 
     * this dependency is on a derived parameter (not just coords) of the i-th SubspaceState AND if any 
     * derived params of the i-th SubspaceState depend on shared parameters at all (or just coords of i-th subspace state)*/
    std::vector< std::vector< std::pair<size_t, bool>> > dependencies;
    //TODO make static? 

    Float weight = 1;

    bool isInitialized = false;

    int sharedDependencyMaxDepth = 1;

    void add(const std::shared_ptr<SubspaceState>& s) { state.push_back(s); } 

    /* Warning: no smart dependency resolution here! That's fine for initialization.
     * During stepping this function will not be called. */
    void eval() {
        for (int i = 0; i < sharedDependencyMaxDepth; ++i) { 
            for (auto&& subspace : state) { 
                subspace->eval(shared); 
                update_shared(shared, subspace);
            }
        }
    }

    //void init(const std::vector<std::string>& sharedNames, int sharedDependenceMaxDepth = 1) {
    void init() {
        //
        std::cout << "\nResolving dependencies for state with " << state.size() << " likelihoods \n";
        //initShared(sharedNames);
        //initShared();
      
        /* given the dependencies requested of the subspace states, collect for each one who depends on it */ 
        dependencies.resize(state.size(), {});
        for (size_t i = 0; i < state.size(); ++i) { 

             //also, build list of sharedNames while we are at it TODO lilkely remove as well as init_shared function which was only one needing it  
            //sharedNames.insert(sharedNames.end(), states[i]->requestedSharedNames.begin(), states[i]->requestedSharedNames.end());

            for (const auto& n : state[i]->requestedSharedNames) {

                bool found = false;

                std::cout <<"Searching for " << n << ".\n";
                for (size_t j = 0; j < state.size(); ++j) {
                    
                    /* not interested in itself */
                    if (i == j) continue; 

                    const auto& l = state[j];

                    /* does l contain the requested parameter? */
                    if (auto c = l->getCoords(n)) {

                        found = true;

                        std::cout <<"Found in Likelihood " << j << "\n";

                        /* acquire initial value & build the map of all shared parameters. 
                         * not necessary when the name has already been requested by someone else, but it does not hurt either
                         * so don't check that */
                        shared[n] = *c;

                        /* take note that state[j] offers this shared parameter */
                        state[j]->offeredSharedNames.insert(n);

                        /* add this dependency of i-th state on l to l (index j), if this is news */

                        int alreadyPresent = -1;
                        bool flag = l->derivedOnShared && l->isDerived(n);

                        for (size_t m = 0; m < dependencies[j].size(); ++m) { 
                            if (dependencies[j][m].first == i)
                                alreadyPresent = int(m);
                        }
                        if (alreadyPresent == -1) {  
                            dependencies[j].push_back(std::make_pair(i, flag));
                            std::cout << "Dependency of Likelihood " << i << " on " << j << " registered\n";
                        }
                        else {   /* may still need to raise the strength of this known depenendency if flag is true (and if it wasn't before) */ 
                            dependencies[j][alreadyPresent].second = dependencies[j][alreadyPresent].second || flag;
                            std::cout << "Dependency of Likelihood " << i << " on " << j << " updated.\n";
                        }

                        break;
                    }
                }

                if (!found)
                    std::cout << "Shared name was not found in any subspace state.\n" << std::endl;
            } // end go through all requested shared names of states[i]
        } // end go through all states i 



        for (size_t i = 0; i < state.size(); ++i) { 
            for (auto& [k, flag] : dependencies[i]) { 
                std::cout << "Likelihood " << k << " depends on likelihood " << i;
                if (flag) 
                    std::cout << " (strongly)\n";
                else
                    std::cout << " (weakly)\n";
            }
        }

            
        /* now can evalute likelihoods */ 
        eval();

        /* remember we did this */
        isInitialized = true;
    }

    /* Warning: this is just a getter! Make sure everything is evaluated. */
    Float loglikelihood() {
        Float l = 0;
        for (auto&& subspace : state) 
            l += subspace->loglike; 
        return l;
    }

    void step_random_subspace(pcg32& rng, ProposalRecord& rec) {

        /* sample a subspace state... */
        do { 
            rec.whichHead = std::min(size_t(rng.nextFloat() * state.size()), state.size()-1);
          /* ... but don't take it if there are no coordinates to step, that is, the coords are all derived (if any) */
        } while (state[rec.whichHead]->getCoords().size() == state[rec.whichHead]->nDerived);  

        if (verbose) 
            std::cout << "Stepping in Likelihood " << rec.whichHead << ".\n";
        step_given_subspace(rng, rec);
      
    }

    /* here proposals are generated for the rec.which'st SubspaceState in state, as well as all dependent Substates.
     * Evalutation graph of dependent SubspaceStates is automatically generated. */
    void step_given_subspace(pcg32& rng, ProposalRecord& rec) {

        /* the actual step in coordiante space */
        auto [headptr, propasymratio] = state[rec.whichHead]->step(rng, shared); 

        rec.shared_prop = shared; 
        rec.prop = {};
        rec.prop[rec.whichHead] = headptr;
        rec.propasymratio = propasymratio;

        /* headptr points to a copy of the rec.which'st SubspaceState. 
         * It has changed coordinates, likelihood and possibly derived params. Need to re-evaluate dependent (on coordinates or derived params of head) 
         * SubspaceStates to proceed. 
         * Note that these dependent SubspaceStates will *not* change their own coordinates, only their likelihoods and possibly some derived parameters (I). 
         * The latter are important: While nothing will further depend on their updated likelihoods, some SubspaceState could 
         * further depend on these derived parameters (II) (including the original SubspaceState head!). 
         * Luckily, the flag in the list of dependencies signals precisely the combination of (I) and (II).
         * We will find a complete std::map<size_t, std::shared_ptr<SubspaceState>> mapping the index of a subspace state to the pointer to its modified copy,
         * which forms the proposal rec.prop. 
         * For evaluation, complicated dependencies may make it necesarry to evaluate the same SubspaceState multiple times. 
         * For example, when A steps, a derived param of B may depends on A's coords. But A may depend on a derived param of B, forcing re-evaluation of A!  
         * To resolve this, we will build a schedule of type std::list<index, pointers to SubspaceStates> (same signature as the map,
         * but ordered and can repeat entries) that will finally be evaluated in order. [std::list can grow while iterating, std::vector would crash doing so]
         * We assume that there are no circular dependencies of derived parameters (of coordinates is allowed). That is, if B depends on derived parameters of A, then A cannot depend on a derived parameter of B (or of C, with C depending on a derived parameter of B, etc.) 
         * (Note this would mean the total likelihood is implicit. We would get an infinite loop, corresponding to trying to find the likelihood as a fixed point of a set of nonlinear equations...) */

        if (verbose) 
            std::cout << "Constructing schedule.\n";

        std::list<std::pair<size_t, std::shared_ptr<SubspaceState>>> schedule{};
        schedule.push_back(std::make_pair(rec.whichHead, headptr));

        auto it = schedule.begin();

        /* to track if there is some infinite loop */
        const size_t MAX_DEPTH = 1000;
        size_t depth = 1;

        /* build the call schedule */
        while (it != schedule.end()) {

            /* go through relevant dependencies */

            auto& currentDependencies = dependencies[it->first];

            for (size_t i = 0; i < currentDependencies.size(); ++i) {

                /* if dependency flag is set, that is, currentDependencies[i].first referes to a SubspaceState that 
                 * depends on a *derived* parameter of the likelihood pointed at by "it", and "it" does update its derived params based on shared ones,
                 * OR if we have actually changed coordinates of "it", that is, if "it" is still = head, which has indeed stepped, we must update further */
                if (currentDependencies[i].second || it == schedule.begin() )  {

                    size_t which = currentDependencies[i].first;
                    /* if the dependent likelihood one appears for the first time in this endeavor...*/
                    if (rec.prop.find(which) == rec.prop.end()) {
                        /* ...prepare its modification by making a copy...*/
                        auto cp = state[which]->copy();
                        /* ...and remember this copy.*/
                        rec.prop[which] = cp;
                    }
                    /* by using the map in rec.prop, we access the copy that has just been made
                     * or that has already been there */
                    schedule.push_back(std::make_pair(which, rec.prop[which]));
                }

                /* else, depenecy can be ignored */
            }

            /* continue finding dependencies of the next dependency... */
            ++it; ++depth;

            if (depth > MAX_DEPTH)  { 
                std::cout << "WARNING: Breaking what looks like infinite dependency loop of derived parameters between different likelihood parts.\n";
                break;
            }
        }

        if (verbose) { 
            std::cout<< "\nSchedule is:\n";
            for (auto& task : schedule)
                std::cout << task.first << " " << & (*(task.second)) << "\n";
        }
        /* evalution phase: work through the schedule */

        /* head's likelihood is already up-to-date after stepping (since this can often be optimized), 
         * but we still need to update shared params before going on */
        update_shared(rec.shared_prop, schedule.begin()->second);

        for (auto it = ++ (schedule.begin()); it != schedule.end(); ++it) { 

            it->second->eval(rec.shared_prop);
            update_shared(rec.shared_prop, it->second);
        }

        rec.deltaloglike = 0;
        for (auto it = rec.prop.begin(); it != rec.prop.end(); ++it)
            rec.deltaloglike += -state[it->first]->loglike + it->second->loglike;

        //std::cout << "previous loglike is " << state[rec.which]->loglike  << " and new one is " << ret.first->loglike << "\n";

    }

    void accept(const ProposalRecord& rec) {
        shared = rec.shared_prop;
        
        /* this includes the coordinates, stepsize and loglike already */
        for (auto it = rec.prop.begin(); it != rec.prop.end(); ++it)
            state[it->first] = it->second;

        weight = rec.weight;
    }

    void correctStepsizeCorrectionFac(const ProposalRecord& rec, Float fac) {
        state[rec.whichHead]->stepsizeCorrectionFac *= fac;
    }

    const std::vector<Float> getCoords(const std::string& name) const {
        for (const auto& substate : state)
            if (auto r = substate->getCoords(name))
                return *r;
        return std::vector<Float>{};
    }

private:

    //std::vector<std::string> sharedNames;

    /* if a shared name is found in from, its value is updated the the coords in from */
    void update_shared(SharedParams& s, const std::shared_ptr<SubspaceState> from) {

        if (verbose) 
            std::cout << "Updating relevant shared parameters: ";

        for (const auto& n : from->offeredSharedNames) {
            s[n] = *(from->getCoords(n));
            if (verbose) 
                std::cout << n << " ";
        }
        if (verbose) 
            std::cout << ".\n";
        //for (auto it = s.begin(); it != s.end(); ++it) {

            //auto coords = from->getCoords(it->first);

            //if (coords) 
                //it->second = *coords;
        //}
    }

    /* slow function to make shared params hash map, trying to find each shared name in any SubspaceState */
    //void init_shared() {

        //for (auto& n : sharedNames) {

            //bool found = false;

            //for (auto& l : state) {

                //if (auto c = l->getCoords(n)) {
                    //found = true;
                    //shared[n] = *c;
                    //break;
                //}
            //}

            //if (!found)
                //std::cout << "Shared name was not found in any subspace state.\n" << std::endl;
        //}
    //}



};

/* Target is a wrapper around State that adds user-defined weighing. 
 * The standard Markov Chain is going to run on Target, not State, but e.g. simmulated annealing runs on State directly. */

class SimpleTarget {

public:

    void set_posterior(std::shared_ptr<State> s) { 
        
        state = s; 

        /* initialization is crucial, but user may have not done it (if no shared parameters are needed) - do it here */
        if (!state->isInitialized)
            state->init();
            //state->init({});
            //
        state->weight = weight();

    } 

    /* Important note for any derived class implementation: if replaceBy given, use this instead of the replaceWhich'st subspace state in state.state vector of pointers to subspacestates */
    virtual Float weight(std::shared_ptr<SubspaceState> replaceBy = nullptr, size_t replaceWhich = 0) const {
        return 1;
        //std::shared_ptr<SubspaceState> calc = state->state[0];
        //if (replaceBy != nullptr)
            //calc = replaceBy;

        //Float x = (*calc->getCoords("position"))[0];

        //return std::exp(-x*x/2);
    }

    virtual Float beta(Float time) { 
        return 1;
    }

    void step(pcg32& rng, ProposalRecord& rec, Float time, bool isSubspaceRandom) {

        if (isSubspaceRandom)
            state->step_random_subspace(rng, rec);
        else
            state->step_given_subspace(rng, rec);

        Float weightOld = weight(); //TODO: should be fine to read if from state.
        Float weightNew = weight(rec.prop[rec.whichHead], rec.whichHead);

        rec.deltaloglike += (std::log(weightNew) - std::log(weightOld));
        rec.deltaloglike *= beta(time);
        rec.weight = weightNew;

    }


    virtual ~SimpleTarget() {} 

    std::shared_ptr<State> state;

};

class CoolingTarget : public SimpleTarget {

public:

    Float slope;

    CoolingTarget(Float coolingSlope) : slope(coolingSlope) {} 

    Float beta(Float time) override { 
        return std::exp(time*slope);
    }

    virtual ~CoolingTarget() {} 

};

/*
        START_NAMED_TIMER("Chains") 
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, nChains, 1),
                [&](tbb::blocked_range<size_t> range) {
                    for (size_t chain = range.begin(); chain < range.end(); ++chain) {
                    //for (int chain = 0; chain < nChains; ++chain) {
                        std::cout << "Chain " << chain << " started.\n"; 
                    */
class MetropolisChain {

public:

    int id; 
    pcg32 rnd;
    std::shared_ptr<SimpleTarget> target;

    MetropolisChain(std::shared_ptr<SimpleTarget> target, int chainid) : id(chainid), rnd(0, chainid), target(std::move(target)) {

    }

    void run(int nSamples, int nBurnin, int nAdjust, int thinning = 100) {

        int nAccept = 0;
        std::cout << "Adjusting step sizes\n";
        int nBar = 60;
        int adjustbar = 0;
        int progressbar = 0;

        START_NAMED_TIMER("Adjustment");
        for (int i = 0; i < nSamples+nAdjust; ++i)  {

            auto singleStep = [&](ProposalRecord& rec, int& accept, bool isSubspaceRandom)  { 
                target->step(rnd, rec, Float(i)/nAdjust, isSubspaceRandom);

                if (verbose)
                    std::cout << "Delta loglike (incl. weight and temp) " << rec.deltaloglike << ".\n";
                if ( rnd.nextFloat() < rec.propasymratio*std::exp(rec.deltaloglike) ) { 
                    target->state->accept(rec);
                    ++accept;
                    if (verbose)
                        std::cout << "Accepted.\n";
                }
            };

            ProposalRecord rec; 

            if (i == nAdjust)  { 
                std::cout << "\n\nFinished adjustments. Now sampling.\n";
                nAccept = 0;
                STOP_TIMER;
                START_NAMED_TIMER("Sampling");
            }

            if (i < nAdjust) {

                if (adjustbar < int(nBar*Float(i)/nAdjust)) { 
                    std::cout << "#" << std::flush;
                    ++adjustbar;
                }

                nAccept = 0;
                singleStep(rec, nAccept, true);

                int nRepeat = 100;
                if (verbose)
                    nRepeat = 10;

                for (int j = 0; j < nRepeat-1; ++j)  
                    singleStep(rec, nAccept, false);

                Float acceptRate = Float(nAccept)/nRepeat; 

                /* maps 0.234 to 1, 0+ to zero and 1 to 2, smoothly */
                auto rate2corr = [](Float x) { 
                    return (1+0.726484*x*x*x*x)/(0.82051 + 0.0427315/(x+0.0001));
                };

                target->state->correctStepsizeCorrectionFac(rec, rate2corr(acceptRate));

                if (verbose)
                    std::cout << "Adjusted step size to " << rec.prop.begin()->second->stepsizeCorrectionFac << ".\n";

            }
            else {

                if (progressbar < int(nBar*Float(std::max(0, i-nAdjust))/nSamples)) { 
                    std::cout << "#" << std::flush;
                    ++progressbar;
                }
                singleStep(rec, nAccept, true);


            }
            if ((i > nBurnin) && (i % thinning) == 0) 
                samples.push_back(std::make_shared<State>(*(target->state)));
        }
        STOP_TIMER;

        std::cout << "\n\nChain " << id << " finished, acceptance rate was " << Float(nAccept)/(nBurnin+nSamples) << ".\n";

    }

    //void cool();

#if PY == 1

    py::array_t<Float> getSamples(const std::string& coordName) {

        if (samples.size() == 0) { 
            std::cout << "Requested samples, but none are present\n";
            return {};
        }
    
        auto data0 = samples[0]->getCoords(coordName); 

        py::array_t<Float> ret({/*times number of chains */samples.size(), data0.size()});

        for (size_t i = 0; i < samples.size(); ++i)  { 
            
            int idx = data0.size() * (0*id*samples.size() + i);
            auto data = samples[i]->getCoords(coordName); 
            std::memcpy(&(ret.mutable_data()[idx]), &(data[0]), sizeof(Float)*data0.size());

        }

        return ret;
    }

    py::array_t<Float> getWeights() {

        if (samples.size() == 0) { 
            std::cout << "Requested weights, but none are present\n";
            return {};
        }
    
        py::array_t<Float> ret(/*times number of chains */samples.size());

        for (size_t i = 0; i < samples.size(); ++i)  { 
            
            ret.mutable_data()[i] = 1/samples[i]->weight;
        }

        return ret;
    }
    
    py::array_t<Float> getLoglikes() {

        if (samples.size() == 0) { 
            std::cout << "Requested weights, but none are present\n";
            return {};
        }
    
        py::array_t<Float> ret(/*times number of chains */samples.size());

        for (size_t i = 0; i < samples.size(); ++i)  { 
            
            ret.mutable_data()[i] = samples[i]->loglikelihood();
        }

        return ret;
    }
#endif


protected:

    std::vector< std::shared_ptr<State> > samples;

};

