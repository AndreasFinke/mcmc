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

    HAS_STEP

private:

    std::vector<std::vector<Float>> datapoints;

};

/* this mock likelihood only samples x and y, but is a standard gaussian in 4d in x,y,z and w.
 * To test working with shared parameters that are stepped elsewhere, z is part of likelihood C.
 * To test dependency on derived parameters of other likelihoods, w^2 is derived in a gaussian likelihood D for w 
 * that is stepped there and the likelihood for w is split into a part in D and the part in A using wsq.
 * Finally, we introduce a mock depency on a derived parameter x-y 
 * that drops out in our expression for the likelihood ((x+y)(x-y) = x^2 - y^2),
 * The extra complication here is that this derived parameter, computed in likeihood B, depends on the coordinates of A! */

class A : public  SubspaceState {

public:

    A() : SubspaceState({"x and y", "xpy"}, 1, false) {

        setCoords( {{1, 1}, {2}} );
        requestedSharedNames.push_back("z");
        requestedSharedNames.push_back("wsq");
        requestedSharedNames.push_back("xmy");

    }

    void eval(const SharedParams& shared) override {

        /* this demos how to access the parameters quickly given their indices in coords array */
        Float x = coords[0][0];
        Float y = coords[0][1];
        /* note the use of a reference (&) type "Float&" to be able to modify this derived paramter by modifying xpy*/
        Float& xpy = coords[1][0];

        /* eval has to compute all derived parameters */
        xpy = x + y;

        /* this demos how to request shared parameters */
        Float z = shared.at("z")[0];
        Float wsq = shared.at("wsq")[0];
        Float xmy = shared.at("xmy")[0];
            
        /* eval's main task to compute loglike. note the missing factor of 2 for wsq, the rest sits in D!*/ 
        loglike = -(x*x + 3*y*y + 2*z*z + xpy*xmy + wsq)/4;

    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        /* only modify (and return) the copy. If derived class functions have to be accessed, use auto newstate = dynamic_pointer_cast<A>(copy());! */
        auto newstate = copy();
       
        /* this demos how to access the coordinates quickly given their indices in the coords array. 
         * the access function is needed because newstate is another instance and coords is hidden for us */
        Float& x = newstate->getCoords()[0][0];
        Float& y = newstate->getCoords()[0][1];

        x += stepsizeCorrectionFac*(rnd.nextFloat()-0.5);
        y += stepsizeCorrectionFac*(rnd.nextFloat()-0.5);

        newstate->eval(shared);
        return Proposal{newstate, 1};
    }

    /* don't forget to write this line here */
    HAS_STEP
};

/* this likelihood needs "x and y" vector from another and computes a derived parameter, x - y, called "xmy" */
class B : public  SubspaceState {

public:

    /* not how 1 indicates that the last 1 parameters in the list are derived. Thus, all in this case. */ 
    B() : SubspaceState({"xmy"}, 1, true) {
        setCoords( {{99}} ); //anything... code has to be smart enough to update this given the real values later 
        requestedSharedNames.push_back("x and y");
    }

    void eval(const SharedParams& shared) override {

        auto xy = shared.at("x and y");
        Float x = xy[0];
        Float y = xy[1];

        getCoordsAt("xmy")[0] = x - y; 

        /*If you only want to map some shared parameters to derived parameters just don't touch loglike, 
         * or for clarity set it to 0 */
        loglike = 0;
    }

    HAS_STEP
    /*note how all parameters of B are derived (there is one parameter and one derived parameter)
     * in this case, step does not have to be implemented, and we do not need to write HAS_STEP */
};

/* this likelihood needs "x and y" vector from another and computes a derived parameter, x - y, called "xmy" */
class C : public  SubspaceState {

public:

    /* note how not mentioning properties of derived parameters (numbers and if they depend on shared) defaults to not having any */
    C() : SubspaceState({"z"}) {
        setCoords( {{-1}} ); //anything... code has to be smart enough to update this given the real values later 
    }


    /*note how since eval is trivial it does not have to be implemented. This will hardly happen in practice though */

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = copy();
      
        /* let's demo a second way besides getCoords()[0][0] to access z, in case we forgot the how-maniest parameter it is:
         * request a parameter vector by name*/
        Float& z = newstate->getCoordsAt("z")[0];

        z += stepsizeCorrectionFac*(rnd.nextFloat()-0.5);

        return Proposal{newstate, 1};
    }

    HAS_STEP

    /*note how all parameters of B are derived (there is one parameter and one derived parameter)
     * in this case, step does not have to be implemented, and we do not need to write HAS_STEP */
};

class D : public SubspaceState { 
public:
    D() : SubspaceState({"w", "wsq"}, 1, false) {

        setCoords( {{-1},{-1}} );

    }

    void eval(const SharedParams& shared) override {

        /* this demos how to access the parameters quickly given their indices in coords array */
        Float w = coords[0][0];

        coords[1][0] = w*w;

        loglike = -w*w/4;

    }

    Proposal step(pcg32& rnd, const SharedParams& shared) const override {

        auto newstate = copy();
       
        Float& w = newstate->getCoords()[0][0];

        w += stepsizeCorrectionFac*(rnd.nextFloat()-0.5);

        newstate->eval(shared);
        return Proposal{newstate, 1};
    }

    /* don't forget to write this line here */
    HAS_STEP
};

class MyState : public State {

public:

    MyState() { 

        auto l1 = std::make_shared<MyLike1>();

        state.push_back(l1);

        init();

    }

};




