//
// Created by Yifei Li on 3/15/21.
// Email: liyifei@csail.mit.edu
//

#ifndef OMEGAENGINE_BACKWARDTASKSOLVER_H
#define OMEGAENGINE_BACKWARDTASKSOLVER_H

#include "../engine/Macros.h"
#include "../engine/Shader.h"
#include "../engine/Camera.h"
#include "../engine/VertexArrayObject.h"
#include "../simulation/Simulation.h"
#include "OptimizationTaskConfigurations.h"
#include "../engine/UtilityFunctions.h"
#include <time.h>
#include "OptimizeHelper.h"

class BackwardTaskSolver {
public:
    static void
    optimizeLBFGS(std::shared_ptr<Simulation> system, OptimizeHelper& helper,
                  int FORWARD_STEPS,  int demoNum, bool isRandom,
                  int srandSeed, const std::function<void(const std::string &)> &setTextBoxCB);



    static void setWindSim2realInitialParams(Simulation::ParamInfo &paramGroundtruth,
                                             Simulation::BackwardTaskInformation &taskInfo, std::shared_ptr<Simulation> system);

    static void setDemoSceneConfigAndConvergence(std::shared_ptr<Simulation> system, int demoNum, Simulation::BackwardTaskInformation &taskInfo);

    static void resetSplineConfigsForControlTasks(int demoNum, std::shared_ptr<Simulation> system,
                                                  Simulation::ParamInfo &paramGroundtruth);

 

    static void
    setLossFunctionInformationAndType(LossType &lossType, Simulation::LossInfo &lossInfo, std::shared_ptr<Simulation> system,
                                      int demoNum);


    static void
    setInitialConditions(int demoNum, std::shared_ptr<Simulation> system,
                         Simulation::ParamInfo &paramGroundtruth, Simulation::BackwardTaskInformation &taskInfo);

    static void
    solveDemo(std::shared_ptr<Simulation> system, const std::function<void(const std::string &)> &setTextBoxCB,
              int demoNum, bool isRandom, int srandSeed);

    static OptimizeHelper getOptimizeHelper(std::shared_ptr<Simulation> system, int demoNum);

    static std::shared_ptr<OptimizeHelper> getOptimizeHelperPointer(std::shared_ptr<Simulation> system, int demoNum);

};


#endif //OMEGAENGINE_BACKWARDTASKSOLVER_H
