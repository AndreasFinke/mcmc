/*
    Copyright (c) 2020 Andreas Finke <andreas.finke@unige.ch>

    All rights reserved. Use of this source code is governed by a modified BSD
    license that can be found in the LICENSE file.
*/


#pragma once

#include "mcmc.h"
#include <memory>

class MyLike1 : public  SubspaceState {

public:

    MyLike1() : SubspaceState({"position", "max"}, 1, false), datapoints{ {2, 2, 0}, {-2,-2,0}} {

        setCoords( {{1, 1, 1}, {1}} );

    }

    //MyLike1(const MyLike1& rhs) : SubspaceState(rhs), datapoints{rhs.datapoints} {
    //}

    ~MyLike1() {} 

    void eval(const SharedParams& shared) override {

        loglike = 0;

        for (size_t i = 0; i < datapoints.size(); ++i) 
            for (size_t j = 0; j < datapoints[i].size(); ++j)
                loglike -= (datapoints[i][j] - coords[names["position"]][j])*(datapoints[i][j] - coords[names["position"]][j])*Float(0.5); 

        coords[names["max"]][0] = std::max( std::max( coords[0][0], coords[0][1]), coords[0][2] ); 

        if (verbose) 
            std::cout<< "MyLike1 loglike evaluated to " << loglike << "\n.";
    }

    //Proposal step_impl(pcg32& rnd, const SharedParams& shared) const override {
    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = copy();

        //std::cout << "steppin " << stepsizeCorrectionFac << std::endl;
        for (auto& p : newstate->getCoords()[names.at("position")])
            p += 10*stepsizeCorrectionFac*(rnd.nextFloat()-0.5);

        //for (int i = 0; i < getCoords()[0].size(); ++i)
            //newstate->getCoords()[0][i] += 100*rnd.nextFloat();
            
            
        newstate->eval(shared);
        return Proposal{newstate, 1};

    }

    std::shared_ptr<SubspaceState> copy() const override {
        return std::shared_ptr<SubspaceState>(new MyLike1(*this));
    }

private:

    std::vector<std::vector<Float>> datapoints;

};


class MyState : public State {

public:

    MyState() { 

        auto l1 = std::make_shared<MyLike1>();

        state.push_back(l1);

        init();

    }

};




