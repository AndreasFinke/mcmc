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
#include "covid.h"

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
    py::class_<DiseaseData>(m, "DiseaseData")
        .def(py::init<const py::array_t<Float>, const py::array_t<Float>>())
        .def_readwrite("initialBetaMild", &DiseaseData::initialBetaMild)
        .def_readwrite("initialBetaHigh", &DiseaseData::initialBetaHigh)
        .def_readwrite("initialDelay", &DiseaseData::initialDelay)
        .def_readwrite("fixBehaviorInAdvance", &DiseaseData::fixBehaviorInAdvance);
    py::class_<DiseaseParams>(m, "DiseaseParams")
        .def(py::init<>())
        .def_readwrite("probAsymp", &DiseaseParams::probAsymp)
        .def_readwrite("probLethal", &DiseaseParams::probLethal)
        .def_readwrite("probSerious", &DiseaseParams::probSerious)
        .def_readwrite("probICUIfSerious", &DiseaseParams::probSerious)
        .def_readwrite("timeIncub", &DiseaseParams::timeIncub)
        .def_readwrite("timeIncubSigma", &DiseaseParams::timeIncubSigma)
        .def_readwrite("timeMildDuration", &DiseaseParams::timeMildDuration)
        .def_readwrite("timeMildDurationSigma", &DiseaseParams::timeMildDurationSigma)
        .def_readwrite("timeMildToSerious", &DiseaseParams::timeMildToSerious)
        .def_readwrite("timeMildToSeriousSigma", &DiseaseParams::timeMildToSeriousSigma)
        .def_readwrite("timeSeriousToRecovered", &DiseaseParams::timeSeriousToRec)
        .def_readwrite("timeSeriousToRecoveredSigma", &DiseaseParams::timeSeriousToRecSigma)
        .def_readwrite("timeSeriousToDeath", &DiseaseParams::timeSeriousToDeath)
        .def_readwrite("timeSeriousToDeathSigma", &DiseaseParams::timeSeriousToDeathSigma)
        .def_readwrite("probLethalDailyWhenSeriousUntreated", &DiseaseParams::probLethalDailyWhenSeriousUntreated);
    py::class_<AvgDiseaseTrajectory>(m, "AvgDiseaseTrajectory")
        .def(py::init<const DiseaseParams&>())
        .def("getIncubating", &AvgDiseaseTrajectory::getIncubating)
        .def("getAsymptomatic", &AvgDiseaseTrajectory::getAsymptomatic)
        .def("getMild", &AvgDiseaseTrajectory::getMild)
        .def("getMildlyInfectious", &AvgDiseaseTrajectory::getMildlyInfectious)
        .def("getHighlyInfectious", &AvgDiseaseTrajectory::getHighlyInfectious)
        .def("getSerious", &AvgDiseaseTrajectory::getSerious)
        .def("getDead", &AvgDiseaseTrajectory::getDead)
        .def("getRecovered", &AvgDiseaseTrajectory::getRecovered)
        .def_readwrite("incubating", &AvgDiseaseTrajectory::incubating)
        .def_readwrite("asymptomatic", &AvgDiseaseTrajectory::asymptomatic)
        .def_readwrite("mild", &AvgDiseaseTrajectory::mild)
        .def_readwrite("infectiousMild", &AvgDiseaseTrajectory::infectiousMild)
        .def_readwrite("infectiousHigh", &AvgDiseaseTrajectory::infectiousHigh)
        .def_readwrite("serious", &AvgDiseaseTrajectory::serious)
        .def_readwrite("dead", &AvgDiseaseTrajectory::dead)
        .def_readwrite("recovered", &AvgDiseaseTrajectory::recovered);
    py::class_<DiseaseSpread, SubspaceState, std::shared_ptr<DiseaseSpread>>(m, "DiseaseSpread")
        .def(py::init<const DiseaseData&, const DiseaseParams&, int, double, double, int, size_t>());

    py::class_<A, SubspaceState, std::shared_ptr<A>>(m, "A")
        .def(py::init<>());
    py::class_<B, SubspaceState, std::shared_ptr<B>>(m, "B")
        .def(py::init<>());
    py::class_<C, SubspaceState, std::shared_ptr<C>>(m, "C")
        .def(py::init<>());
    py::class_<D, SubspaceState, std::shared_ptr<D>>(m, "D")
        .def(py::init<>());

    py::class_<SmoothnessPrior, SubspaceState, std::shared_ptr<SmoothnessPrior>>(m, "SmoothnessPrior")
        .def(py::init<const std::string&, Float, Float>());
    py::class_<State, std::shared_ptr<State>>(m, "State")
        .def(py::init<>())
        .def("add", &State::add)
        .def("init", &State::init)
        .def("loglike", &State::loglikelihood)
        .def("force_bounds", &State::force_bounds)
        .def("getAll", &State::getAll)
        .def("test", &State::test)
        .def_readwrite("sharedDependencyMaxDepth", &State::sharedDependencyMaxDepth);
    py::class_<MyState, State, std::shared_ptr<MyState>>(m, "MyState")
        //.def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def(py::init<>());
    py::class_<SimpleTarget, std::shared_ptr<SimpleTarget>>(m, "SimpleTarget")
        .def(py::init<>())
        .def("set_posterior", &SimpleTarget::set_posterior);
    py::class_<CoolingTarget, SimpleTarget, std::shared_ptr<CoolingTarget>>(m, "CoolingTarget")
        .def(py::init<Float, Float>());
        //.def("add_state", &Target::add_state);
    py::class_<AdvCoolingTarget, SimpleTarget, std::shared_ptr<AdvCoolingTarget>>(m, "AdvCoolingTarget")
        .def(py::init<Float, Float>())
        .def_readwrite("maxPeriodLength", &AdvCoolingTarget::maxPeriodLength)
        .def_readwrite("minOscillations", &AdvCoolingTarget::minOscillations)
        .def_readwrite("defaultHeatCapacity", &AdvCoolingTarget::defaultHeatCapacity);
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
        .def("getMean", &MetropolisChain::getMean)
        .def("getSamples", &MetropolisChain::getSamples)
        .def("getWeights", &MetropolisChain::getWeights)
        .def("getLoglikes", &MetropolisChain::getLoglikes)
        .def_readwrite("computeMean", &MetropolisChain::computeMean)
        .def_readwrite("recordSamples", &MetropolisChain::recordSamples);
    py::class_<GradientDecent>(m, "GradientDecent")
        //.def(py::init<py::array_t<double>, py::array_t<double>, int>())
        .def(py::init<std::shared_ptr<State>, Float>())
        //.def("decent", &GradientDecent::decent)
        .def("adaptive_gd", &GradientDecent::adaptive_gd)
        .def("accelerated_adaptive_gd", &GradientDecent::accelerated_adaptive_gd)
        .def("nesterov_accelerated_gd", &GradientDecent::nesterov_accelerated_gd)
        .def("perturb", &GradientDecent::perturb)
        .def_readwrite("learningRate", &GradientDecent::learningRate);
        //.def_readwrite("Lsmooth", &Sampler::Lsmooth);
    //py::class_<Dist>(m, "Dist")
        //.def(py::init<int, float, float>())
        //.def("pdf", &Dist::pdf);
}

#endif
