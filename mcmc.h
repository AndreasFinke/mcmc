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
#include <set>
#include <list>
#include <type_traits>
#include <assert.h>

#include <tbb/tbb.h>
#include <cmath>
#include <sstream>
#include <fstream>

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

#include "enoki/special.h"
#include "timer.h"
#include "pcg32.h"

#if PY == 1

#include <pybind11/numpy.h>

namespace py = pybind11;

#endif

using Float = double;

template <typename T> T toNormal(T uniform) {
    return Float(M_SQRT2) * enoki::erfinv(Float(2) * uniform - Float(1));
}

template <typename T> void bound(T& val, T lower, T upper) {
    val = T(0.5)*(val-lower)/(upper-lower);
    val = T(2)*std::abs(val-std::round(val));
    val = lower + val*(upper-lower);
}

#define HAS_STEP std::shared_ptr<SubspaceState> copy() const override {\
    return std::make_shared<std::remove_const_t<std::remove_reference_t<decltype(*this)>>>(*this);}

using SharedParams = std::map<std::string, std::vector<Float>>;

class State; 

class SubspaceState {

friend class State;
friend class GradientDecent;

public:

    Float loglike = 0;
    Float stepsizeCorrectionFac = 1;
    bool derivedOnShared;
    std::set<std::pair<size_t, size_t>> fixed = {};

    SubspaceState(std::vector<std::string>&& coordNames, bool derivedParamsDependOnSharedParams = false, size_t nDerived = 0) :  derivedOnShared(derivedParamsDependOnSharedParams), nDerived(nDerived) {  
        for (size_t i = 0; i < coordNames.size(); ++i) {
            names[coordNames[i]] = i;
        }
    } 
    /* derived class constructor must call this! */
    void setCoords(std::vector<std::vector<Float>> init) {
        coords = init;
    }


    /* the second number is the pij/pji correction factor for the MCMC */
    using Proposal = std::pair<std::shared_ptr<SubspaceState>, Float>;

    /* derived class needs to implement these */
    virtual std::shared_ptr<SubspaceState> copy() const = 0;
    virtual void eval(const SharedParams& shared) { 
        loglike = 0;
    }
    //virtual Proposal step_impl(pcg32& rnd, const SharedParams& shared) const = 0;
    virtual Proposal step(pcg32& rnd, const SharedParams& shared) const { 
        auto newstate = copy();
        return Proposal{newstate, 1};
    }

    virtual std::vector<Float> sampleInitialConditions(pcg32& rnd) {
        return {};
    }

    virtual std::vector<Float> getInitialConditions() {
        return {};
    }

    virtual void setInitialConditions(const std::vector<Float>& ics) {
    }

    virtual void force_bounds() {}
    virtual ~SubspaceState() {};

    inline auto& getCoords()  { return coords; }
    inline const auto& getCoords() const { return coords; }

    inline std::optional<std::vector<Float>> getCoords(const std::string name) const {
        auto it = names.find(name);
        if (it != names.end()) {
            return coords[it->second]; 
        }
        else
            return {};
    }

    inline std::vector<Float>& getCoordsAt(const std::string name) {
        return coords[names.at(name)]; 
    }

    std::map<std::string, std::vector<Float>> getAll() {
        std::map<std::string, std::vector<Float>> ret = {};
        for (auto [name, idx] : names) 
            ret[name] = coords[idx];
        return ret;
    }

    inline bool isDerived(const std::string name) const {
        auto it = names.find(name);
        if (it != names.end()) {
            return it->second >= names.size()-nDerived; 
        }
        else
            return {};
    }

    inline bool isFixed(std::pair<size_t, size_t> index) const {
        return fixed.find(index) != fixed.end();
    }

    auto get_names() const {
        return names;
    }


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

};

class SmoothnessPrior : public SubspaceState {

public:
    SmoothnessPrior(const std::string& functionName, Float smoothnessScale, Float scale) : SubspaceState({}), L(scale), Lsmooth(smoothnessScale) {

        requestedSharedNames.push_back(functionName);

        /* no coords to set */
    }

    Float L, Lsmooth; 
    
    void eval(const SharedParams& shared) override {

        auto f = shared.at(requestedSharedNames[0]);
        
        //if (!f) {
            //std::cout << "SmoothnessPrior cannot evaluate: " << requestedSharedNames[0] << " is not part of the shared parameters.\n";
            //return;
        //}
        Float dx = L/f.size();  
        loglike = 0;
        for (size_t i = 1; i < f.size()-1; ++i) { 
            Float dd = f[i+1] + f[i-1] - 2*f[i]; 
            loglike -= dd*dd/dx/dx/dx; /*one dx cancels from the integration*/

        }

        loglike *= Lsmooth*Lsmooth/L;
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
    std::vector< std::vector<Float>  > initialConditions;

    /* for each, i-th, SubspaceState in state, we keep a corresponding i-th element in dependencies:
     * a list of the other SubspaceStates that depend on the i-th SubspaceStates, together with a flag if 
     * this dependency is on a derived parameter (not just coords) of the i-th SubspaceState AND if any 
     * derived params of the i-th SubspaceState depend on shared parameters at all (or just coords of i-th subspace state)*/
    std::vector< std::vector< std::pair<size_t, bool>> > dependencies;
    //TODO make static? 

    Float weight = 1;

    bool isInitialized = false;
    bool isEvaluated = false;

    int sharedDependencyMaxDepth = 10;

    void add(const std::shared_ptr<SubspaceState>& s) { state.push_back(s); } 

    void addCoordsOf(const std::shared_ptr<State>& s) {
        for (size_t i = 0; i < state.size(); ++i) {
            for (size_t j = 0; j < state[i]->getCoords().size(); ++j) {
                for (size_t k = 0; k < state[i]->getCoords()[j].size(); ++k) { 
                    state[i]->getCoords()[j][k] += s->state[i]->getCoords()[j][k];
                }
            }
        }
    }
    void fma(const std::shared_ptr<State>& s, Float fac) {
        for (size_t i = 0; i < state.size(); ++i) {
            for (size_t j = 0; j < state[i]->getCoords().size(); ++j) {
                for (size_t k = 0; k < state[i]->getCoords()[j].size(); ++k) { 
                    state[i]->getCoords()[j][k] += fac*s->state[i]->getCoords()[j][k];
                }
            }
        }
    }
    void assign(const std::shared_ptr<State>& s) {
        for (size_t i = 0; i < state.size(); ++i) {
            *(state[i]) = *(s->state[i]);
        }
    }
    void multCoordsBy(Float a) {
        for (size_t i = 0; i < state.size(); ++i) {
            for (size_t j = 0; j < state[i]->getCoords().size(); ++j) {
                for (size_t k = 0; k < state[i]->getCoords()[j].size(); ++k) { 
                    state[i]->getCoords()[j][k] *= a;
                }
            }
        }
    }

    void perturb(pcg32& rnd, Float a) {
        for (size_t i = 0; i < state.size(); ++i) {
            for (size_t j = 0; j < state[i]->getCoords().size(); ++j) {
                for (size_t k = 0; k < state[i]->getCoords()[j].size(); ++k) { 
                    state[i]->getCoords()[j][k] += (rnd.nextFloat()-.5f)*state[i]->getCoords()[j][k]*a;
                }
            }
        }
    }

    /* note this modifies the initialConditions saved in State, and instead of relying on SubspaceState::sampleIntitialConditions to actually also set them internally (it should) it sets explicitely */
    std::vector< std::vector<Float> > sampleInitialConditions(pcg32& rnd) {
        initialConditions.resize(0);
        for (size_t i = 0; i < state.size(); ++i) {
            initialConditions.push_back(state[i]->sampleInitialConditions(rnd));
        }
        isEvaluated = false;
        setInitialConditions(initialConditions);
        return initialConditions;
    }

    /* assuming they are set internally, this works even after State::sampleInitialConditions. And if they have changed internally it's an update request */
    std::vector< std::vector<Float> > getInitialConditions() {
        initialConditions.resize(0);
        for (size_t i = 0; i < state.size(); ++i) {
            initialConditions.push_back(state[i]->getInitialConditions());
        }
        return initialConditions;
    }
    void setInitialConditions(const std::vector< std::vector<Float> >& ics) {
        assert(ics.size() == state.size());
        initialConditions = ics;
        for (size_t i = 0; i < state.size(); ++i) {
            state[i]->setInitialConditions(ics[i]);
        }
        isEvaluated = false;
        eval();
    }

    void set_stepsizes(const std::vector<Float>& sz) {
        for (size_t i = 0; i < state.size(); ++i) 
            state[i]->stepsizeCorrectionFac = sz[i];
    }
    std::vector<Float> get_stepsizes() {
        std::vector<Float> ret = {};
        for (size_t i = 0; i < state.size(); ++i) 
            ret.push_back(state[i]->stepsizeCorrectionFac);
        return ret;
    }

    std::map<std::string, std::vector<Float>> get_all() { 
        std::map<std::string, std::vector<Float>> ret = {};
        for (size_t i = 0; i < state.size(); ++i) { 
            auto r = state[i]->getAll();
            ret.insert(r.begin(), r.end());
        }
        return ret;
    }

    void force_bounds() {
        for (size_t i = 0; i < state.size(); ++i) 
            state[i]->force_bounds();
    }

    std::shared_ptr<State> deep_copy() {
        State ret(*this); //shallow copy as needed when collecting samples 
        for (auto& s : ret.state) 
            s = s->copy(); 
        return std::make_shared<State>(ret);
    }

    /* Warning: no smart dependency resolution here! That's fine for initialization.
     * During stepping this function will not be called. */
    void eval() {
        for (int i = 0; i < sharedDependencyMaxDepth; ++i) { 
            for (auto&& subspace : state) { 
                subspace->eval(shared); 
                update_shared(shared, subspace);
            }
        }
        isEvaluated = true;
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

                std::cout <<"Searching for " << n << "...";
                for (size_t j = 0; j < state.size(); ++j) {
                    
                    /* not interested in itself */
                    if (i == j) continue; 

                    const auto& l = state[j];

                    /* does l contain the requested parameter? */
                    if (auto c = l->getCoords(n)) {

                        found = true;

                        std::cout <<" - found in Likelihood " << j << ".\n";

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

                if (!found) { 
                    std::cout << "Shared name was not found in any subspace state.\n" << std::endl;
                    throw std::string("Missing shared param");
                }
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
        std::cout << "First evaluation of complete likelihood...\n";
        eval();
        std::cout << "... done.\n";

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

    /* assuming only the changed element of the "state" array of SubspaceStates has changed, evaluate only what is needed. 
     * This is the same algorithm as above in the step function, but here we do not need and therefore do not want to copy the 
     * relevant dependent SubspaceStates but modify them in place. It is easiest to write a new, similar function because of slight differences throughout... for comments see above */
    void eval_graph(size_t changed) {
        //std::map<size_t, std::shared_ptr<SubspaceState>> dependent;

        if (verbose) 
            std::cout << "Constructing schedule (eval_graph).\n";

        std::list<std::pair<size_t, std::shared_ptr<SubspaceState>>> schedule{};
        schedule.push_back(std::make_pair(changed, state[changed]));

        auto it = schedule.begin();
        const size_t MAX_DEPTH = 1000;
        size_t depth = 1;
        while (it != schedule.end()) {
            auto& currentDependencies = dependencies[it->first];
            for (size_t i = 0; i < currentDependencies.size(); ++i) 
                if (currentDependencies[i].second || it == schedule.begin() )  {
                    size_t which = currentDependencies[i].first;
                    schedule.push_back(std::make_pair(which, state[which]));
                }
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
        /* here we cannot assume the first SubspaceState is already evaluated since we did not step */
        for (auto it = schedule.begin(); it != schedule.end(); ++it) { 
            it->second->eval(shared);
            update_shared(shared, it->second);
        }
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

};

/* Target is a wrapper around State that adds user-defined weighing. 
 * The standard Markov Chain is going to run on Target, not State, but e.g. simmulated annealing runs on State directly. */

class Target {

public:

    void set_posterior(std::shared_ptr<State> s) { 
        
        try {
            if (!s->isInitialized)
                s->init();
            if (!s->isEvaluated) { 
                s->eval();
                std::cout << "Target::set_posterior received unevaluated state and forced evaluation.\n";
            }

            state = s; 
            state->weight = weight();

        }
        catch (...) {
            std::cout << "set_posterior failed. Fix Your setup and try again." << std::endl;
        }  

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

    Float logprobability(Float time) {
        return std::log(weight()) + beta(time)*state->loglikelihood();
    }

    virtual bool step(pcg32& rnd, ProposalRecord& rec, Float time, bool isSubspaceRandom, int& nAccept) {

        if (state == nullptr) { 
            rec.prop = {};
            return false;
        }

        if (isSubspaceRandom)
            state->step_random_subspace(rnd, rec);
        else
            state->step_given_subspace(rnd, rec);

        Float weightOld = weight(); //TODO: should be fine to read if from state.
        Float weightNew = weight(rec.prop[rec.whichHead], rec.whichHead);

        rec.deltaloglike += (std::log(weightNew) - std::log(weightOld));
        rec.deltaloglike *= beta(time);
        rec.weight = weightNew;

        if (rec.prop.size() == 0) {
            if (verbose)
                std::cout << "No proposal received...\n";
            return false;
        }
        if (verbose)
            std::cout << "Delta loglike (incl. weight and temp) " << rec.deltaloglike << ".\n";
        if ( rnd.nextFloat() < rec.propasymratio*std::exp(rec.deltaloglike) ) { 
            state->accept(rec);
            ++nAccept;
            if (verbose)
                std::cout << "Accepted.\n";
        }
        return true;
    }

    virtual std::shared_ptr<Target> deep_copy() {
        Target t(*this);
        t.state = state->deep_copy();
        return std::make_shared<Target>(t);
    }

    virtual ~Target() {} 

    std::shared_ptr<State> state = nullptr;

};

class TempTarget : public Target {

public:

    TempTarget(Float T = 1) : bet(1/T) {} 


    Float bet; 

    Float beta(Float time) override { 
        return bet;
    }


    std::shared_ptr<Target> deep_copy() override {
        TempTarget t(*this);
        t.state = state->deep_copy();
        return std::make_shared<TempTarget>(t);
    }

    virtual ~TempTarget() {} 

};

class CoolingTarget : public Target {

public:

    Float slope, Tinit;

    CoolingTarget(Float coolingSlope, Float Tinit) : slope(coolingSlope), Tinit(Tinit) {} 

    Float beta(Float time) override { 
        return std::exp(time*slope)/Tinit;
    }
    
    std::shared_ptr<Target> deep_copy() override {
        CoolingTarget t(*this);
        t.state = state->deep_copy();
        return std::make_shared<CoolingTarget>(t);
    }

    virtual ~CoolingTarget() {} 

};

class AdvCoolingTarget : public Target {

    std::vector<Float> energies;
    Float energy_old, relaxationTime; 
    Float T, DeltaT; 
    Float slope;
    bool first = true;

    int startindex = 0;

    bool measure(Float& energy) { 

        energies.push_back(-state->loglikelihood());

        /* actually half a period below */
        int currentPeriodLength = 0;

        int currentOscillations = 0;
        int lastSign = 1;
        bool isEquil = false;

        for (size_t i = startindex+1; i < energies.size(); ++i) {

            /* increment current period length. If too big, reset counter of continous short periods currentOscillations 
             * and start looking for that from here next time */
            if (++currentPeriodLength > maxPeriodLength) {
                currentPeriodLength = 0;
                currentOscillations = 0;
                startindex = i;
            }
            /* sign of delta energy has changed. increase counter and resent period length counter.  */
            if ( (energies[i]-energies[i-1])*lastSign < 0 ) {
                lastSign = -lastSign;
                currentOscillations++;
                currentPeriodLength = 0;
            }
            /* we think this is equilibrium, since engergy fluctuates both ways quickly for a long time */
            if( currentOscillations >= minOscillations) {
                std::cout <<"Thinking equilibirum at i = " << i << " of " << energies.size() << " saved energies.\n";
                isEquil = true;
                relaxationTime = i - startindex;
            }
        }

        if (isEquil) {
            //Float avg = 0;
            //for (size_t i = startindex; i < energies.size(); ++i) 
                //avg += energies[i];
            //avg /= (energies.size() - startindex);
            //
            energies.erase(energies.begin()+startindex);
            std::sort(energies.begin(), energies.end());
            Float median = energies[energies.size()/2]; 
            energies.resize(0);
            //energy = avg;
            energy = median;
        }

        /* */
        return isEquil;
    }

public:

    int maxPeriodLength = 6;
    int minOscillations = 20;
    Float defaultHeatCapacity = -1;

    AdvCoolingTarget(Float coolingSlope, Float Tinit) : slope(coolingSlope), T(Tinit) {
        DeltaT = T/4;
    } 
    
    Float beta(Float time) override { 
        return 1/T;
    }

    bool step(pcg32& rnd, ProposalRecord& rec, Float time, bool isSubspaceRandom, int& nAccept) override {

        bool ret = Target::step(rnd, rec, time, isSubspaceRandom, nAccept);

        /* only consider accepted states, as samples from the equilibrium or non-equilibrium */ 
        if (nAccept == 1) { 
       
            /* keep record current energy until system is relaxed, in which case relaxation will be updated, energy_new will be set, 
             * and true returned */
            Float energy_new; 
            bool relaxed = measure(energy_new);

            if (relaxed) { 

                /* if this is the second time that it has relaxed (or later) update DeltaT */ 
                if (!first) { 
                    Float heatCapacity = -(energy_new - energy_old)/DeltaT;

                    std::cout << "heatcap " << heatCapacity << "\n";
                    if (heatCapacity < 0) { 
                        //std::cout << "Negative heat capacity due to nonconverged estimation. Continuing... \n"; 
                        if (defaultHeatCapacity > 0) 
                            heatCapacity = defaultHeatCapacity;
                    }

                    DeltaT = sgn(heatCapacity)*slope*T/(relaxationTime*std::sqrt(std::abs(heatCapacity)));

                    std::cout << "T = " << T << ", deltaT = " << DeltaT << ", (" << DeltaT/T*100 << "percent)\n";

                }

                first = false;
                
                /* equilibrium has been reached, so cool more! */
                T -= DeltaT;
                
                energy_old = energy_new;
            }
        
        }

        return ret; 
    }

    std::shared_ptr<Target> deep_copy() override {
        AdvCoolingTarget t(*this);
        t.state = state->deep_copy();
        return std::make_shared<AdvCoolingTarget>(t);
    }
    virtual ~AdvCoolingTarget() {} 

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

template <class T>
class ChainManager;

class MetropolisChain {

    friend class ChainManager<MetropolisChain>;

public:

    bool computeMean = false;
    bool recordSamples = true;
    bool writeSamplesToDisk = false;
    int id; 
    pcg32 rnd;
    std::shared_ptr<Target> target;

    Float weight;

    MetropolisChain(std::shared_ptr<Target> targetArg, int chainid) : id(chainid), rnd(0, chainid), target(targetArg), weight(1) {
        if (verbose)
            std::cout << "Constructing chain\n";
        mean = target->state->deep_copy();
    }

    void run(int nSamples, int nBurnin, int nAdjust, int thinning = 100) {

        int nAccept = 0;
        std::cout << "Adjusting step sizes\n";
        int nBar = 60;
        int adjustbar = 0;
        int progressbar = 0;

        std::ofstream file;
        if (writeSamplesToDisk) { 
            std::stringstream filename; 
            filename << "samples" << id << ".txt";
            file.open(filename.str());
            file << "chainweight " << weight << "\n\n";
        }

        START_NAMED_TIMER("Adjustment");

        for (int i = 0; i < nSamples+nAdjust; ++i)  {

            ProposalRecord rec; 

            if (i == nAdjust)  { 
                std::cout << "\n\nFinished adjustments.\n";
                for (auto & s : target->state->state)
                    std::cout << s->stepsizeCorrectionFac << "\n";
                std::cout << "Now sampling.\n";
                nAccept = 0;
                STOP_TIMER;
                START_NAMED_TIMER("Sampling");
            }

            if ((i >= nBurnin) && (i % thinning) == 0) {
                if (i < nAdjust) 
                    loglikes.push_back(target->logprobability(Float(i)/nAdjust));
                else
                    loglikes.push_back(target->logprobability(Float(i-nAdjust)/nSamples));

                ics.push_back(target->state->getInitialConditions());

                if (writeSamplesToDisk) { 
                    auto data = target->state->get_all();

                    file << "weight " << 1/target->state->weight << " loglike " << target->state->loglikelihood() << "\n";
                    for (auto& [name, vec] : data) {
                        file << name << " ";

                        for (Float x : vec)
                            file << x << " ";
                        file << "\n";
                    }
                    file << "\n";
                }
                if (recordSamples)
                    samples.push_back(std::make_shared<State>(*(target->state))); 
            }

            if ((i > nBurnin) && computeMean)
                mean->addCoordsOf(target->state);

            //if (!recordSamples) { [> noticed slowdown almost 2x when *not* saving samples. This seems to 
                                     //be due to many immediate deletes of subspace states when share_ptrs go 
                                     //out of scope. Avoid this by collecting some and deleting them at once. 
                                     //On my system (Mac Catalina) this resolved the issue. Update: may not resolve the issue for small enough nGarbage
                                     //that it actually deletes. Usually this is needed to save memory, which is the whole point of not recording samples...*/ 
                //static std::vector<std::shared_ptr<State>> garbage;
                //const int nGarbage = 50;
                //static int curr = 0;
                //if (curr < nGarbage) 
                    //garbage.push_back(std::make_shared<State>(*(target->state)));
                //else if (curr == nGarbage) {
                    //curr = -1;
                    //garbage = {};
                //}
                //curr++;
            //}
            if (i < nAdjust) {

                if (adjustbar < int(nBar*Float(i)/nAdjust)) { 
                    std::cout << "#" << std::flush;
                    ++adjustbar;
                }

                nAccept = 0;
                if (!target->step(rnd, rec, Float(i)/nAdjust, true, nAccept)) 
                    return; 
                //if (!singleStep(rec, nAccept, true))
                    //return; 

                int nRepeat = 20;
                if (verbose)
                    nRepeat = 10;

                for (int j = 0; j < nRepeat-1; ++j) {
                    int oldaccept = nAccept;
                    target->step(rnd, rec, Float(i)/nAdjust, false, nAccept);
                    //if (nAccept > oldaccept)
                    //std::cout << rec.deltaloglike <<" (" << target->state->loglikelihood() << ")\n";
                }

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
                
                target->step(rnd, rec, Float(i-nAdjust)/nSamples, true, nAccept);
            }


        }
        STOP_TIMER;

        if (computeMean) 
            mean->multCoordsBy(Float(1)/(nAdjust+nSamples-nBurnin));

        std::cout << "\n\nChain " << id << " finished, acceptance rate was " << Float(nAccept)/(nBurnin+nSamples) << ".\n";

    }


    void reevaluate(std::shared_ptr<State> evalstate, int nBurnin) {

        if (!recordSamples && !writeSamplesToDisk) {
            std::cout << "Re-evaluation of chain " << id << " is not requested to record samples in chain nor write them to disk. Aborting.\n";
            return;
        }
        if (ics.size() == 0) { 
            std::cout << "Re-evaluation of chain " << id << " impossible: No initial conditions have been recorded.\n";
            return;
        }
        if (samples.size() > 0) {
            std::cout << "Re-evaluation of chain " << id << " despite saved samples. Overwriting... \n";
            samples = {};
        }
        if (nBurnin >= ics.size()) { 
            std::cout << "Requested re-evaluation with burnin as big as original run or bigger. Aborting.\n";
            return;
        }
       
        std::vector<decltype(ics)::value_type>(ics.begin()+nBurnin, ics.end()).swap(ics);
        std::vector<decltype(loglikes)::value_type>(loglikes.begin()+nBurnin, loglikes.end()).swap(loglikes);

        if (writeSamplesToDisk) {
            std::stringstream filename; 
            filename << "samples" << id << ".txt";
            std::ofstream file(filename.str());
            file << "chainweight = " << weight << "\n\n";
            for (size_t i = 0; i < ics.size(); ++i) {

                evalstate->setInitialConditions(ics[i]);
                target->set_posterior(evalstate);

                file << "weight " << 1/target->state->weight << " loglike " << target->state->loglikelihood() << "\n";

                auto data = target->state->get_all();

                for (auto& [name, vec] : data) {
                    file << name << " ";

                    for (Float x : vec)
                        file << x << " ";
                    file << "\n";
                }
                file << "\n";

                if (recordSamples) /* here we cannot rely on copies being made of the changed subspacestates due to chain stepping - need to copy right here */
                    samples.push_back(target->state->deep_copy()); 
            }
        }
        else {  /* recordSamples is true if writeSamplesToDisk is false, because of previous test*/ 
            for (size_t i = 0; i < ics.size(); ++i) {
                evalstate->setInitialConditions(ics[i]);
                target->set_posterior(evalstate);

                /* here we cannot rely on copies being made of the changed subspacestates due to chain stepping - need to copy right here */
                samples.push_back(target->state->deep_copy()); 
            }
        }
    }
#if PY == 1

    py::array_t<Float> get_mean(const std::string& coordName) {

        if (!computeMean)
            std::cout << "Requested mean, but MetropolisChain::computeMean is false\n";

        auto data = mean->getCoords(coordName); 

        py::array_t<Float> ret(data.size());
            
        std::memcpy(ret.mutable_data(), &(data[0]), sizeof(Float)*data.size());

        return ret;
    }
    py::array_t<Float> get_samples(const std::string& coordName) {

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

    py::array_t<Float> get_weights() {

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
    
    py::array_t<Float> get_loglikes() {

        if (loglikes.size() == 0) { 
            std::cout << "Requested loglikes, but none are present\n";
            return {};
        }
    
        py::array_t<Float> ret(/*times number of chains */loglikes.size());

        for (size_t i = 0; i < loglikes.size(); ++i)  { 
            
            ret.mutable_data()[i] = loglikes[i];
        }

        return ret;
    }
#endif


protected:

    std::vector< std::shared_ptr<State> > samples;
    std::vector<Float> loglikes;
    std::vector< std::vector< std::vector<Float> > > ics;
    std::shared_ptr<State> mean;

};

template<class ChainType> 
class ChainManager  { 
public:
    ChainManager(std::shared_ptr<Target> target, size_t nChain) : target(target), nChain(nChain), stepsizes{}, chains{}, chainICs{}  {

        pcg32 rnd(0, 0);
        /*ICs, probability */

        std::cout << "\nConstructing " << nChain << " random inital states. " << std::endl; 
        
        while (chainICs.size() < nChain) {

            /* sample some new ICs and evaluate posterior */
            target->state->sampleInitialConditions(rnd);
            //target->state->eval();
            target->set_posterior(target->state);

            /* fix t = 0 for time dependent temperature targets */
            Float prob = target->logprobability(0);

            if ( !std::isnan(prob) && !std::isinf(prob) ) {
                std::cout << "Logprobabilty of state " << chainICs.size() << " is " << prob << "\n";

                /* forget about initial probability. Assuming burnin will be used and chain will mix */
                chainICs.push_back(std::make_pair(target->state->initialConditions, 1)); 
            //std::cout << target->state->loglikelihood() << " " << prob << "\n";

            }

        }

        /* no way to know so far - can obtain later in run_all_adjust */
        stepsizes = std::vector<Float>(target->state->state.size(), Float(1));
    }
    ChainManager(std::shared_ptr<Target> target, size_t nChainReservoir, size_t nChain) : target(target), nChain(nChain), stepsizes{}, chains{}, chainICs{}  {

        pcg32 rnd(0, 0);
        /*ICs, probability */
        std::vector<std::pair<std::vector<std::vector<Float>>, Float>> trialChainICs;

        std::cout << "\nConstructing a discrete distribution of " << nChainReservoir << " random inital states. " << std::endl; 
        
        while (trialChainICs.size() < nChainReservoir) {

            /* sample some new ICs and evaluate posterior */
            target->state->sampleInitialConditions(rnd);
            target->state->eval();

            /* fix t = 0 for time dependent temperature targets */
            Float prob = target->logprobability(0);

            if ( !std::isnan(prob) && !std::isinf(prob) ) {
                trialChainICs.push_back(std::make_pair(target->state->initialConditions, prob)); 
            //std::cout << target->state->loglikelihood() << " " << prob << "\n";

            }

        }
        stepsizes = std::vector<Float>(target->state->state.size(), Float(1));
        bootstrap(rnd, trialChainICs);
    }

    ChainManager(std::shared_ptr<ChainType> generator, std::shared_ptr<Target> targetRHS, size_t nChain) : target(targetRHS), nChain(nChain), stepsizes{}, chains{}, chainICs{}  { 

        pcg32 rnd(0, 0);
        std::vector<std::pair<std::vector<std::vector<Float>>, Float>> trialChainICs;
        //auto generatorTarget = std::make_shared<TempTarget>(generatorTemp);
        //generatorTarget->set_posterior(target->state);
        //generator = std::make_shared<ChainType>(generatorTarget, 0);
        //generator->recordSamples = false;
        //std::cout << "Creating these from " << nChainReservoir*generatorSkip + nGeneratorBurnin << " samples of a generator chain at temperature " << generatorTemp << ". 
        //generator->run(nChainReservoir*generatorSkip, nGeneratorBurnin, nGeneratorBurnin, generatorSkip);
        
        if (!generator->recordSamples)
            std::cout << "Re-evaluating chain samples for their likelihood given the current target. This is as many times quicker as the original chain as the thinning.\n";

        for (size_t i = 0; i < generator->loglikes.size(); ++i) {
            Float oldlogprob = generator->loglikes[i];
            /* if samples have not been saved fully, re-evaluate original loglikelihood with new target - I gave up on inverting oldlogprob with its target (the member of the generator chain) to get the original loglike and put that through the new target just to 
             * save some a fraction of the time the full generator chain took (assuming stong thinning). Targets include weights that depend on whatever... perhaps even derived parameters? so 
             * we should re-evaluate */
           
            if (generator->recordSamples) {
                target->set_posterior(generator->samples[i]);
            }
            else  { 

                /* re-evaluate loglikelihood with new target - I gave up on inverting oldlogprob with its target (the member of the generator chain) to get the original loglike and put that through the new target just to 
                 * save some a fraction of the time the full generator chain took (assuming stong thinning). Targets include weights that depend on whatever... perhaps even derived parameters? so 
                 * we should re-evaluate */
                target->state->setInitialConditions(generator->ics[i]);
                //target->state->eval(); //already done in other functions. 
                /* this is assigning target's state to itself, but also computing the new weight! */
                target->set_posterior(target->state);
            }

            Float newlogprob = target->logprobability(0); 
            trialChainICs.push_back(std::make_pair(generator->ics[i], newlogprob - oldlogprob));

            //std::cout << generator->ics[i][0][0] << " " << generator->ics[i][0][1] << "\t"; 
        }

        stepsizes = generator->target->state->get_stepsizes();
        bootstrap(rnd, trialChainICs);
    }

    ChainManager(std::vector<ChainType> generators, std::shared_ptr<Target> targetRHS, size_t nChain) : target(targetRHS), nChain(nChain), stepsizes{}, chains{}, chainICs{}  { 

        if (generators.size() == 0) {
            std::cout << "Generator chains passed to ChainMananger do not contain any chains. Aborting construction. \n"; 
        }
        else {
            pcg32 rnd(0, 0);
            std::vector<std::pair<std::vector<std::vector<Float>>, Float>> trialChainICs;
            
            if (!generators[0].recordSamples)
                std::cout << "Re-evaluating chain samples for their likelihood given the current target. This is as many times quicker as the original chain as the thinning.\n";

            for (auto& generator : generators) { 
                for (size_t i = 0; i < generator.loglikes.size(); ++i) {

                    /* higher chain weight is a sample weight, equivalent to LOWER probability of the distribution (less likely that chain goes there, and indeed, 
                     * we want to say that we have only one chain going here, but should have "weight" chains. The lower probability 
                     * does increase the chance of picking this state as expected, since we need to subtract this number below! */
                    Float oldlogprob = generator.loglikes[i] - std::log(generator.weight);
                   
                    if (generator.recordSamples) {
                        target->set_posterior(generator.samples[i]);
                    }
                    else  { 

                        /* re-evaluate loglikelihood with new target - I gave up on inverting oldlogprob with its target (the member of the generator chain) to get the original loglike and put that through the new target just to 
                         * save some a fraction of the time the full generator chain took (assuming stong thinning). Targets include weights that depend on whatever... perhaps even derived parameters? so 
                         * we should re-evaluate */
                        target->state->setInitialConditions(generator.ics[i]);
                        //target->state->eval(); //already done in other functions. 
                        /* this is assigning target's state to itself, but also computing the new weight! */
                        target->set_posterior(target->state);
                    }

                    Float newlogprob = target->logprobability(0); 
                    trialChainICs.push_back(std::make_pair(generator.ics[i], newlogprob - oldlogprob));

                }

            }

            stepsizes = generators[0].target->state->get_stepsizes();
            bootstrap(rnd, trialChainICs);
        }
    }

    std::vector<Float> stepsizes;

    void run_all(size_t nSteps, size_t thinning, Float stepsizefac, int nThread) {
        tbb::task_scheduler_init init(nThread);

        for (auto& s : stepsizes) 
                s *= stepsizefac;

        chains = {};

        for (size_t i = 0; i < chainICs.size(); ++i) {

            /* we want separate targets (and states) for parallel chains */
            std::shared_ptr<Target> tar = target->deep_copy();

            /* calls State::eval() */
            tar->state->setInitialConditions(chainICs[i].first);
            tar->state->set_stepsizes(stepsizes);

            tar->set_posterior(tar->state);
            
            chains.push_back(ChainType(tar, i+1));

            std::cout << "Initialized chain " << i << " with true loglikelihood = " << tar->state->loglikelihood() << "\n";
            chains[i].recordSamples = false;
            chains[i].weight = chainICs[i].second;
        }

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, chainICs.size(), 1),
                [&](tbb::blocked_range<size_t> range) {
                

                for (size_t i = range.begin(); i < range.end(); ++i) {
                //for (size_t i = 0; i < chains.size(); ++i) {
                    std::cout << "Started chain " << i << "." << std::endl; 


                    chains[i].run(nSteps, 0, 0, thinning);

                }
            }
        );

        std::cout << "Chain runs completed." << std::endl << std::endl;

    }
    void run_all_adjust(size_t nSteps, size_t nAdjust, size_t thinning, int nThread) {
        tbb::task_scheduler_init init(nThread);
        chains = {};

        for (size_t i = 0; i < chainICs.size(); ++i) {

            /* we want separate targets (and states) for parallel chains */
            std::shared_ptr<Target> tar = target->deep_copy();

            /* calls State::eval() */
            tar->state->setInitialConditions(chainICs[i].first);

            tar->set_posterior(tar->state);
            
            chains.push_back(ChainType(tar, i+1));

            std::cout << "Initialized chain " << i << " with true loglikelihood = " << tar->state->loglikelihood() << "\n";
            chains[i].recordSamples = false;
            chains[i].weight = chainICs[i].second;
        }

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, chainICs.size(), 1),
                [&](tbb::blocked_range<size_t> range) {
                

                for (size_t i = range.begin(); i < range.end(); ++i) {
                //for (size_t i = 0; i < chains.size(); ++i) {
                    std::cout << "Started chain " << i << "." << std::endl; 


                    chains[i].run(nSteps, 0, nAdjust, thinning);

                }
            }
        );

        stepsizes = chains[0].target->state->get_stepsizes();
        std::cout << "Chain runs completed." << std::endl << std::endl;

    }

    ChainType& get_chain(size_t i) {
        if (chains.size() == 0) {   
            std::cout << "ChainManager::get_chain: Requested chain " << i << "of the manager, but no valid chain is present. call ChainManager::run_chains first!\n";
            throw;
        }
        if (chains.size() <= i) {   
            std::cout << "ChainManager::get_chain: Requested chain " << i << "of the manager, but only up to " << chains.size() << " chains are present. Valid indices up to " <<chains.size() -1 << " only!\n";
            throw;
        }
        return chains[i];
    }

    auto get_all_chains() {
        if (chains.size() == 0) {   
            std::cout << "ChainManager::get_all_chains: Requested chains of the manager, but no valid chain is present. call ChainManager::run_chains first!\n";
        }
        return chains;
    }

    void reevaluate_all(std::shared_ptr<State> state, int nBurnin, bool recordSamples, bool writeSamplesToDisk, int nThread) {
        tbb::task_scheduler_init init(nThread);
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, chainICs.size(), 1),
                [&](tbb::blocked_range<size_t> range) {
                

                for (size_t i = range.begin(); i < range.end(); ++i) {
                //for (size_t i = 0; i < chains.size(); ++i) {
                    std::cout << "Reevaluating chain with id = " << i << "." << std::endl; 
         
                    chains[i].recordSamples = recordSamples;
                    chains[i].writeSamplesToDisk = writeSamplesToDisk;
                    chains[i].reevaluate(state->deep_copy(), nBurnin);
                    

                }
            }
        );
        //
        std::cout << "Reevaluation completed." << std::endl << std::endl;

    }
protected:

    void bootstrap(pcg32& rnd, std::vector< std::pair<std::vector<std::vector<Float>>, Float> >& trialChainICs) { 
        /*the corresponding discrete CDF as a float array*/
        std::vector<double> discreteCDF;
        double total = 0.0;
        /* normalize all probabilities by the same factor, bt shifting log prob such that the best sits at zero. this is to avoid all of the exponentials being rounded to zero for loglikelihoods that may be of the order of -1e7... */
        double maxsofar = -1e20;
        for (size_t i = 0; i < trialChainICs.size(); ++i)
            if (trialChainICs[i].second > maxsofar)
                maxsofar = trialChainICs[i].second;

        std::cout << "Selecting " << nChain << " chains according to the target density." << std::endl; 
        std::cout << "Highest log probability in set: " << maxsofar << "\n";

        for (size_t i = 0; i < trialChainICs.size(); ++i)  { 
            trialChainICs[i].second = std::exp(trialChainICs[i].second - maxsofar);
            total += trialChainICs[i].second; 
        }

        discreteCDF.push_back(trialChainICs[0].second);
        for (size_t i = 1; i < trialChainICs.size(); ++i) 
            discreteCDF.push_back(discreteCDF[i-1] + trialChainICs[i].second);
        discreteCDF.back() = 1.000001*total; //fight rounding errors: has to be larger than largest random number for what comes now 

        std::map<size_t, size_t> old2new; 
        
        for (size_t i = 0; i < nChain; ++i) {
            double x = rnd.nextDouble()*total;
            auto iter = std::upper_bound(discreteCDF.begin(), discreteCDF.end(), x);
            size_t idx = iter - discreteCDF.begin();

            std::cout << "Relative loglikelihood of chain drawn is " << std::log(trialChainICs[idx].second) << "\n";
            /* is it the first time these ICs are sampled? */
            if (old2new.find(idx) == old2new.end()) {
                /* no need to remember the probability of the initial state */
                chainICs.push_back(std::make_pair(std::move(trialChainICs[idx].first), 1)); 
                /* save the index of this chain in the current chainICs array in a map as a function of the index into the trialChainICs */
                old2new[idx] = chainICs.size()-1;  
            }
            else {
                std::cout << "Note: same chain " << old2new[idx] << " that has been picked " << chainICs[old2new[idx]].second << " times was picked again. New chain weight: " << chainICs[old2new[idx]].second + 1 << std::endl;
                chainICs[old2new[idx]].second += 1;
                --i;
            }

        }
        for (size_t i = 0; i < nChain; ++i) {
            std::cout << "\n" << " selected ICs are \n" <<  chainICs[i].first[0][0] << " " << chainICs[i].first[0][1] << "\t"; 
        }


        std::cout << "Selection completed." << std::endl << std::endl; 
    }
    std::vector< std::pair<std::vector<std::vector<Float>>, int>> chainICs;
    std::vector<ChainType> chains = {};
    size_t nChain = 1;
    std::shared_ptr<Target> target;
};


class GradientDecent {

public:

    GradientDecent(const std::shared_ptr<State>& state, Float eps) : state(state), 
    lambda(eps), Lambda(eps), theta(1e20), Theta(1e20), epsilon(eps),
    eta(0) {

        if (!state->isInitialized)
            state->init();

        state_old = state->deep_copy();
        grad = state->deep_copy();

        for (size_t i = 0; i < grad->state.size(); ++i) { 
            //for (size_t j = grad->state.size() - grad->state[i]->nDerived; j < grad->state[i]->coords.size(); ++j) { 
            /* there may be fixed parameters that are not derived, they also need to be zeroed...*/
            for (size_t j = 0; j < grad->state[i]->coords.size(); ++j) { 
                grad->state[i]->coords[j].assign(grad->state[i]->coords[j].size(), Float(0));  
            }
        }

        compute_grad();
        grad_old = grad->deep_copy();
        state->fma(grad, lambda);
        /* initial values are relevant only for assigning to stateHelp_old, which needs to equal state 
         * initially */
        stateHelp = state->deep_copy();
        /* will be assigned to stateHelp anyway but prepare the copy */
        stateHelp_old = state->deep_copy();

    } 

    void compute_grad() {
        Float loglikeold = state->loglikelihood();
        for (size_t i = 0; i < grad->state.size(); ++i) { 
            for (size_t j = 0; j < grad->state[i]->coords.size(); ++j) { 
                /* skip derived parameters */
                if (j >= grad->state[i]->coords.size() - grad->state[i]->nDerived)
                    break;
                for (size_t k = 0; k < grad->state[i]->coords[j].size(); ++k) { 

                    /* only compute gradient of actually free dof */
                    if (grad->state[i]->isFixed(std::make_pair(j, k))) {
                        continue;
                    }

                    state->state[i]->coords[j][k] += epsilon;
                    state->eval_graph(i);
                    Float loglikenew = state->loglikelihood();
                    state->state[i]->coords[j][k] -= epsilon;
                    //state->eval_graph(i);
                    grad->state[i]->coords[j][k] = (loglikenew-loglikeold)/epsilon;
                }
            }
            /* before changing i, reset state to old (including shared params...) */
            state->eval_graph(i);
        }

        //grad->multCoordsBy(learningRate);
        //state->addCoordsOf(grad);
        //state->force_bounds();
    }

    void adaptive_gd(int steps) {


        for (int n = 0; n < steps; ++n) { 

            compute_grad();

            Float xnorm = 0;
            Float gradnorm = 0;
            for (size_t i = 0; i < grad->state.size(); ++i) { 
                for (size_t j = 0; j < grad->state[i]->coords.size(); ++j) { 
                    /* skip derived parameters */
                    if (j >= grad->state[i]->coords.size() - grad->state[i]->nDerived)
                        break;
                    for (size_t k = 0; k < grad->state[i]->coords[j].size(); ++k) { 

                        /* only consider actually free dof */
                        if (grad->state[i]->isFixed(std::make_pair(j, k))) {
                            continue;
                        }

                        Float deltax = state->state[i]->coords[j][k] - state_old->state[i]->coords[j][k];
                        Float deltagrad = grad->state[i]->coords[j][k] - grad_old->state[i]->coords[j][k];

                        xnorm += deltax*deltax;
                        gradnorm += deltagrad*deltagrad;
                    }
                }
            }
            xnorm = std::sqrt(xnorm);
            gradnorm = std::sqrt(gradnorm);

            std::cout << "gradnorm " << gradnorm << " xnorm " << xnorm << " lambda " << lambda << " theta " << theta << "\n";
            Float lambda_new = std::min(std::sqrt(1+theta)*lambda, xnorm*Float(0.5)/gradnorm);

            state_old->assign(state);

            std::cout << "lambda_new " << lambda_new << " coord " << state->state[0]->coords[0][0] << "\n";
            state->fma(grad, lambda_new);
            std::cout << "                 coord " << state->state[0]->coords[0][0] << "\n";

            grad_old->assign(grad);

            theta = lambda_new/lambda;
            lambda = lambda_new;
        
            state->force_bounds();
        }

    }

    void accelerated_adaptive_gd(int steps) {


        for (int n = 0; n < steps; ++n) { 

            compute_grad();

            Float xnorm = 0;
            Float gradnorm = 0;
            for (size_t i = 0; i < grad->state.size(); ++i) { 
                for (size_t j = 0; j < grad->state[i]->coords.size(); ++j) { 
                    /* skip derived parameters */
                    if (j >= grad->state[i]->coords.size() - grad->state[i]->nDerived)
                        break;
                    for (size_t k = 0; k < grad->state[i]->coords[j].size(); ++k) { 

                        /* only consider actually free dof */
                        if (grad->state[i]->isFixed(std::make_pair(j, k))) {
                            continue;
                        }

                        Float deltax = state->state[i]->coords[j][k] - state_old->state[i]->coords[j][k];
                        Float deltagrad = grad->state[i]->coords[j][k] - grad_old->state[i]->coords[j][k];

                        xnorm += deltax*deltax;
                        gradnorm += deltagrad*deltagrad;
                    }
                }
            }
            xnorm = std::sqrt(xnorm);
            gradnorm = std::sqrt(gradnorm);

            std::cout << "gradnorm " << gradnorm << " xnorm " << xnorm << " lambda " << lambda << " theta " << theta << "\n";
            Float lambda_new = std::min(std::sqrt(1+theta)*lambda, xnorm*Float(0.5)/gradnorm);
            Float Lambda_new = std::min(std::sqrt(1+Theta)*Lambda, gradnorm*Float(0.5)/xnorm);

            state_old->assign(state);
            stateHelp_old->assign(stateHelp);

            Float a = std::sqrt(1/lambda);
            Float b = std::sqrt(Lambda);
            Float beta = (a-b)/(a+b);

            stateHelp->assign(state);
            stateHelp->fma(grad, lambda_new);
            std::cout << "lambda_new " << lambda_new << " coord " << state->state[0]->coords[0][0] << "\n";
            state->assign(stateHelp);
            state->fma(stateHelp, beta);
            state->fma(stateHelp_old, -beta);
            std::cout << "                 coord " << state->state[0]->coords[0][0] << "\n";

            grad_old->assign(grad);

            theta = lambda_new/lambda;
            Theta = Lambda_new/Lambda;
            lambda = lambda_new;
            Lambda = Lambda_new;
          
            state->force_bounds();
        }
    }

    void nesterov_accelerated_gd(int steps) {

        for (int n = 0; n < steps; ++n) { 

            compute_grad();

            //Float xnorm = 0;
            //Float gradnorm = 0;
            //for (size_t i = 0; i < grad->state.size(); ++i) { 
                //for (size_t j = 0; j < grad->state[i]->coords.size(); ++j) { 
                    //[> skip derived parameters <]
                    //if (j >= grad->state[i]->coords.size() - grad->state[i]->nDerived)
                        //break;
                    //for (size_t k = 0; k < grad->state[i]->coords[j].size(); ++k) { 

                        //[> only consider actually free dof <]
                        //if (grad->state[i]->isFixed(std::make_pair(j, k))) {
                            //continue;
                        //}

                        //Float deltax = state->state[i]->coords[j][k] - state_old->state[i]->coords[j][k];
                        //Float deltagrad = grad->state[i]->coords[j][k] - grad_old->state[i]->coords[j][k];

                        //xnorm += deltax*deltax;
                        //gradnorm += deltagrad*deltagrad;
                    //}
                //}
            //}
            //xnorm = std::sqrt(xnorm);
            //gradnorm = std::sqrt(gradnorm);

            Float eta_new = Float(0.5)*(1+std::sqrt(1+4*eta*eta));
            Float gamma = (1 - eta)/eta_new;

            state_old->assign(state);
            stateHelp_old->assign(stateHelp);

            stateHelp->assign(state);
            stateHelp->fma(grad, learningRate);
            state->assign(stateHelp);
            state->fma(stateHelp, -gamma);
            state->fma(stateHelp_old, gamma);

            //grad_old->assign(grad);

            eta = eta_new;           
            state->force_bounds();
        }
    }

    void perturb(Float a) {
        state->perturb(rnd, a);
    }

    Float learningRate = 1;

private:

    pcg32 rnd;

    std::shared_ptr<State> state, state_old, stateHelp, stateHelp_old;
    std::shared_ptr<State> grad, grad_old;

    Float epsilon;
    Float lambda, Lambda;
    Float theta, Theta;
    Float eta;

};

class SumConstraint {
public: 

    Float constraint; 
    std::vector<Float>* vals;

    SumConstraint(Float constraint) : constraint(constraint) {} 

    void link(std::vector<Float>* dataptr) { 

        vals = dataptr;
   
        /* check constraint */ 
        Float sum = 0;
        for (auto f : (*vals)) {
            sum += f;
        }
        /* if not satisied, enforce by setting to constant */
        if ( std::fabs((sum-constraint)/constraint) > 0.0001) {
            vals->assign(vals->size(), constraint/vals->size());
            std::cout << "Warning: enforced positive sum contraint destructively.";
        }

    }

    void step(pcg32& rnd, Float stepsize, int& from, int& to, Float& val, Float& volRatio) {

        do { 
            from = std::min(size_t(rnd.nextFloat()*vals->size()), vals->size()-1);
            to   = std::min(size_t(rnd.nextFloat()*vals->size()), vals->size()-1);
            val  = stepsize*rnd.nextFloat();
            //std::cout << " from " << from << " to " << to <<"\n";
        } 
        while ( !moveMass(from, to, val) );

        Float newVol = accessibleStateVol(stepsize);
        Float oldVol = newVol - std::min(stepsize, (*vals)[from]) + std::min(stepsize, (*vals)[to]) - std::min(stepsize, (*vals)[from]+val) - std::min(stepsize, (*vals)[to]-val);

        volRatio = oldVol / newVol;
    }
private: 

    bool moveMass(int from, int to, Float val) { 

        if (from == to) return false;

        if ((*vals)[from] < val) return false;

        (*vals)[to] += val;
        (*vals)[from] -= val;

        return true;
    }

    Float accessibleStateVol(Float stepsize) const { 
        Float ret = 0;
        for (auto f : (*vals)) 
            ret += std::min(f, stepsize);
        return ret;
    }

};
