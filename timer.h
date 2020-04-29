#pragma once

#include <map>
#include <utility>
#include <stack>
#include <string>
#include <iostream>
#include <chrono>

//#define TIMING

struct TinyTimer {
    TinyTimer() : startupTime(get_time()) {}

    ~TinyTimer() { print(); }

    void start(const std::string& name) {
        //std::cout << "Starting timing for " << name << std::endl;
        if (timings.find(name) == timings.end()) {
            //std::cout <<"... for the first time " << std::endl;
            timings[name] = std::make_pair(0, false);
            if (timingStack.empty())  {
                timings[name].second = true;
                //std::cout << "This is a main timer " << std::endl;
            }
        }
        timingStack.push(std::make_pair(name, get_time()));
        //std::cout << "have pushed, stack now has " << timingStack.size() << " elements." << std::endl;
        //currName = name;
        //lastTime = get_time();
    }

    void stop() {
        //std::cout << "Stopping timing for " << timingStack.top().first << std::endl;
        //if (timings.find(timingStack.top().first) == timings.end())
            //std::cout << "wooops" << std::endl;
        timings[timingStack.top().first].first += to_seconds(get_time()-timingStack.top().second);
        timingStack.pop();
        //std::cout << "current stack size is " << timingStack.size() << std::endl;
    }

    void print() {
        std::cout << "Timings: " << std::endl;
        float totalAll =  to_seconds(get_time()-startupTime);
        std::cout << "Total time since startup: " << totalAll << "s" << std::endl;
        float total = 0;
        for (auto const& [key, val] : timings)
            if (val.second)
                total += val.first;
        std::cout << "Total measured time: " << total << "s (" << 100.0f*total/totalAll << "\%)" <<  std::endl;

        for (auto const& [key, val] : timings) 
            std::cout << key << "(" << val.second << "): " << val.first << "s (" << 100.0f*val.first/total << "\% of measured time)" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point get_time() {
        return std::chrono::high_resolution_clock::now();
    };
    //float to_seconds(const std::chrono::high_resolution_clock::time_point& t) { return 0.001f*float(std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count()); }
    float to_seconds(const std::chrono::high_resolution_clock::duration& t) { return float(1e-9)*float(std::chrono::duration_cast<std::chrono::nanoseconds>(t).count()); }
    std::map<std::string, std::pair<float,bool>> timings;
    std::chrono::high_resolution_clock::time_point startupTime; 
    //std::string currName; 

    using NamedTime = std::pair<std::string, std::chrono::high_resolution_clock::time_point>; 
    std::stack<NamedTime> timingStack;

};


#ifdef TIMING 

static TinyTimer timer;

#define START_TIMER timer.start(__PRETTY_FUNCTION__);
#define START_NAMED_TIMER(name) timer.start(name);
#define STOP_TIMER timer.stop();

#else 

#define START_TIMER 
#define START_NAMED_TIMER(name)
#define STOP_TIMER 

#endif 

//#define TIMED_BLOCK( code )  START_TIMER (code) STOP_TIMER
//#define TIMED_NAMED_BLOCK( name , code ) START_NAMED_TIMER(name) (code) STOP_TIMER 
