import time
import torch
import numpy as np
import diffcloth_py as dfc

from pySim.pySim import pySim


def get_default_scene():
    fabric = dfc.FabricConfiguration()
    fabric.clothDimX = 4
    fabric.clothDimY = 4
    fabric.k_stiff_stretching = 70
    fabric.k_stiff_bending = .1
    fabric.gridNumX = 5
    fabric.gridNumY = 5
    fabric.density = 0.24
    fabric.keepOriginalScalePoint = True
    fabric.isModel = False
    fabric.custominitPos = False
    fabric.initPosFile = ""
    fabric.fabricIdx = 0
    fabric.color = np.array([0.9, 0., 0.1])
    fabric.name = "Test"

    scene = dfc.SceneConfiguration()
    scene.fabric = fabric
    scene.orientation = dfc.Orientation.DOWN
    scene.upVector = np.array([0, 1, 0])
    scene.attachmentPoints = dfc.AttachmentConfigs.CUSTOM_ARRAY
    scene.customAttachmentVertexIdx = [(0.0, np.random.randint(25, size=1))]
    scene.trajectory = dfc.TrajectoryConfigs.PER_STEP_TRAJECTORY
    scene.primitiveConfig = dfc.PrimitiveConfiguration.Y0PLANE
    scene.windConfig = dfc.WindConfig.NO_WIND
    scene.camPos = np.array([-21.67, 5.40, -10.67])
    scene.sockLegOrientation = np.array([0., 0., 0.])
    scene.camFocusPointType = dfc.CameraFocusPointType.CLOTH_CENTER
    scene.sceneBbox = dfc.AABB(np.array([-7, -7, -7]), np.array([7, 7, 7]))
    scene.timeStep = 0.01
    scene.stepNum = 500
    scene.forwardConvergenceThresh = 1e-3
    scene.backwardConvergenceThresh = 1e-3
    scene.name = "Test scene"

    return scene


def stepSim(simModule, sim, init, pos):
    start = time.time()
    x = init[0]
    v = init[1]
    x_list, v_list = [x.reshape(-1, 3)], [v.reshape(-1, 3)]
    for step in range(sim.sceneConfig.stepNum):
        p = pos[step]
        x, v = simModule(x, v, p)
        x_list.append(x.reshape(-1, 3))
        v_list.append(v.reshape(-1, 3))

    print("stepSim took {} sec".format(time.time() - start))
    render = True
    if render:
        dfc.render(sim, renderPosPairs=True, autoExit=True)

    return x_list, v_list


def run_step1():
    scene = get_default_scene()
    scene.attachmentPoints = dfc.AttachmentConfigs.NO_ATTACHMENTS

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=1)

    helper = dfc.makeOptimizeHelper("inverse_design")

    sim.forwardConvergenceThreshold =  1e-3

    sim_mod = pySim(sim, helper, True)

    sim.resetSystem()
    state_info_init = sim.getStateInfo()

    pts = state_info_init.x.reshape(-1, 3)

    gp_id = scene.customAttachmentVertexIdx[0][1][0]
    print("gp_id: ", gp_id)

    # Grasp trajectory is filled to prevent crash in stepSim, but values will be ignored
    grasp_point = pts[gp_id].copy()
    grasp_traj = np.tile(grasp_point, (scene.stepNum, 1))
    grasp_traj = torch.tensor(grasp_traj, dtype=torch.float32)

    x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x),
                                  torch.tensor(state_info_init.v)), grasp_traj)

    return x, v


def get_grasp_traj(init, target, length):
    traj = init + (target - init) * np.linspace(0, 1, length)[:, None]
    return torch.tensor(traj)


def run_step2(x_fin1, v_fin1):
    scene = get_default_scene()

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=1)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin1.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    state_info_init = sim.getStateInfo()

    pts = state_info_init.x.reshape(-1, 3)

    gp_id = scene.customAttachmentVertexIdx[0][1][0]
    print("gp_id: ", gp_id)

    grasp_point = pts[gp_id].copy()
    target = grasp_point.copy()
    target[1] = grasp_point[1] + 4
    grasp_traj = get_grasp_traj(grasp_point, target, scene.stepNum)
    # grasp_traj = np.tile(grasp_point, (scene.stepNum, 1))
    # grasp_traj = torch.tensor(grasp_traj, dtype=torch.float32)

    x, v = stepSim(sim_mod, sim, (x_fin1.flatten(), v_fin1.flatten()), grasp_traj)

    return x, v, gp_id


def get_gp2():
    gp2 = np.random.randint(25, size=1)
    if gp2 == gp_id:
        gp2 = get_gp2()
    return gp2


def run_step3(gp1_id, x_fin2, v_fin2):
    scene = get_default_scene()

    gp2_id = get_gp2()
    scene.customAttachmentVertexIdx = [(0.0, [gp1_id, gp2_id])]

    sim = dfc.makeSimFromConf(scene)
    sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
    np.set_printoptions(precision=5)

    dfc.enableOpenMP(n_threads=1)

    helper = dfc.makeOptimizeHelper("inverse_design")
    helper.taskInfo.dL_dx0 = True
    helper.taskInfo.dL_density = False

    sim.forwardConvergenceThreshold = 1e-3

    sim_mod = pySim(sim, helper, True)

    paramInfo = dfc.ParamInfo()
    paramInfo.x0 = x_fin2.flatten().detach().cpu().numpy()

    sim.resetSystemWithParams(helper.taskInfo, paramInfo)

    # sim.resetSystem()
    state_info_init = sim.getStateInfo()
    pts = state_info_init.x.reshape(-1, 3)

    grasp_point_1 = pts[gp1_id].copy()
    grasp1_traj = np.tile(grasp_point_1, (scene.stepNum, 1))
    grasp1_traj = torch.tensor(grasp1_traj, dtype=torch.float32)

    grasp_point_2 = pts[gp2_id].copy()
    target = grasp_point_2.copy()[0]
    target[1] = grasp_point_1[1]
    grasp2_traj = get_grasp_traj(grasp_point_2, target, scene.stepNum)

    grasp_trajs = torch.cat((grasp1_traj, grasp2_traj), dim=1)

    x, v = stepSim(sim_mod, sim, (x_fin2.flatten(), v_fin2.flatten()), grasp_trajs)
    # x, v = stepSim(sim_mod, sim, (torch.tensor(state_info_init.x),
    #                               torch.tensor(state_info_init.v)), grasp_trajs)

    return x, v, (gp1_id, gp2_id)


if __name__=="__main__":
    x1, v1  = run_step1()
    x2, v2, gp_id  = run_step2(x1[-1], v1[-1])
    x3, v4, gp_ids = run_step3(gp_id, x2[-1], v2[-1])
    x1_np = torch.stack(x1).numpy()
    x2_np = torch.stack(x2).numpy()
    x3_np = torch.stack(x3).numpy()
    full_x = np.concatenate((x1_np, x2_np, x3_np))
