/*
    Copyright (c) 2020 Andreas Finke <andreas.finke@unige.ch>

    All rights reserved. Use of this source code is governed by a modified BSD
    license that can be found in the LICENSE file.
*/



#include "test.h"
#include <iostream>

#if PY == 1

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "distfind.h"

namespace py = pybind11;

#endif


#if PY == 0 

int main() {

    pcg32 rng;
    auto t = std::make_shared<SimpleTarget>();

    auto s = std::make_shared<MyState>();

    t->set_posterior(s);

    auto chain = MetropolisChain(t, 0);
    //chain.run(100000, 300, 1000, 5);
    chain.run(100000, 10, 500, 5);

}

#endif

#if PY == 1 

PYBIND11_MODULE(mcmc, m) {
    py::class_<SubspaceState, std::shared_ptr<SubspaceState>>(m, "SubspaceState")
        .def("getNames", &SubspaceState::getNames);
        //.def(py::init<std::map<std::string, int>>());
    py::class_<MyLike1, SubspaceState, std::shared_ptr<MyLike1>>(m, "MyLike1")
        .def(py::init<>());

    py::class_<A, SubspaceState, std::shared_ptr<A>>(m, "A")
        .def(py::init<>());
    py::class_<B, SubspaceState, std::shared_ptr<B>>(m, "B")
        .def(py::init<>());
    py::class_<C, SubspaceState, std::shared_ptr<C>>(m, "C")
        .def(py::init<>());
    py::class_<D, SubspaceState, std::shared_ptr<D>>(m, "D")
        .def(py::init<>());

    py::class_<SmoothnessPrior, SubspaceState, std::shared_ptr<SmoothnessPrior>>(m, "SmoothnessPrior")
        .def(py::init<const std::string&, Float, Float, Float>());
    py::class_<State, std::shared_ptr<State>>(m, "State")
        .def(py::init<>())
        .def("add", &State::add)
        .def("init", &State::init)
        .def_readwrite("sharedDependencyMaxDepth", &State::sharedDependencyMaxDepth);
    py::class_<MyState, State, std::shared_ptr<MyState>>(m, "MyState")
        //.def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def(py::init<>());
    py::class_<SimpleTarget, std::shared_ptr<SimpleTarget>>(m, "SimpleTarget")
        .def(py::init<>())
        .def("set_posterior", &SimpleTarget::set_posterior);
    py::class_<CoolingTarget, SimpleTarget, std::shared_ptr<CoolingTarget>>(m, "CoolingTarget")
        .def(py::init<Float>());
        //.def("add_state", &Target::add_state);
    py::class_<ProbabilityDistributionSamples>(m, "ProbabilityDistributionSamples")
        .def(py::init<py::array_t<Float>, py::array_t<Float>>());
    py::class_<PiecewiseConstantPDF, SubspaceState, std::shared_ptr<PiecewiseConstantPDF>>(m, "PiecewiseConstantPDF")
        .def(py::init<const ProbabilityDistributionSamples&, Float, Float, int>());
    py::class_<GaussianMixturePDF, SubspaceState, std::shared_ptr<GaussianMixturePDF>>(m, "GaussianMixturePDF")
        .def(py::init<const ProbabilityDistributionSamples&, Float, Float, size_t>());
    py::class_<MetropolisChain>(m, "Chain")
        //.def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def(py::init<std::shared_ptr<SimpleTarget>, int>())
        .def("run", &MetropolisChain::run)
        .def("getSamples", &MetropolisChain::getSamples)
        .def("getWeights", &MetropolisChain::getWeights)
        .def("getLoglikes", &MetropolisChain::getLoglikes);
        //.def_readwrite("Lsmooth", &Sampler::Lsmooth);
    //py::class_<Dist>(m, "Dist")
        //.def(py::init<int, float, float>())
        //.def("pdf", &Dist::pdf);
}

#endif
