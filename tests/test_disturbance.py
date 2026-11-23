"""Physical force path: point-wrench equivalence, substeps, coexistence and cleanup."""
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from main import load_config
from sim.sim import build_case, execute_intervals
from sim.controllers import Command, pic, servo
from sim.disturbance import TCPForceEnvironment, apply_tcp_force


class TestDisturbance(unittest.TestCase):
    def case(self, robot='FR3'):
        cfg=load_config('configs/five_obstacle_pic.yaml');cfg['robot']=robot
        cfg['task']['q_start_rad']=None;cfg['sim']['T_max']=.012
        return build_case(cfg)

    def test_world_wrench_equals_jacobian_force_all_models(self):
        from sim.robot import ROBOTS, Robot
        for name,spec in ROBOTS.items():
            r=Robot(spec);m=r.model;q=r.q_home; p,_=r.fk(q);J=r.jac()[:3]
            F=np.array([17.,-23.,9.]);qa=[]
            for mode in ('xfrc','qfrc'):
                d=mujoco.MjData(m);d.qpos[r.qadr]=q;mujoco.mj_forward(m,d)
                if mode=='xfrc':apply_tcp_force(m,d,r.site_body,p,F)
                else:d.qfrc_applied[r.dadr]=J.T@F
                mujoco.mj_forward(m,d);qa.append(d.qacc.copy())
            np.testing.assert_allclose(qa[0],qa[1],rtol=0,atol=1e-9,err_msg=name)

    def test_substep_window_moving_tcp_and_other_loads(self):
        case=self.case();r=case.robot;m=r.model;d=mujoco.MjData(m)
        d.qpos[r.qadr]=case.q_start;d.qvel[:]=.2;mujoco.mj_forward(m,d)
        background=np.arange(m.nbody*6,dtype=float).reshape(m.nbody,6)*.001
        d.xfrc_applied[:]=background
        pu={'t_start_s':.002,'duration_s':.003,'force_N':[2,3,4]}
        env=TCPForceEnvironment(r,[pu]); seen=[];orig=mujoco.mj_step
        def step(m,d):
            p,_=r.fk(d.qpos[r.qadr]);expected=np.cross(p-d.xipos[r.site_body],pu['force_N'])
            k=len(seen);force=np.array(pu['force_N']) if 2<=k<5 else np.zeros(3)
            own=np.r_[force,expected if np.any(force) else np.zeros(3)]
            np.testing.assert_allclose(d.xfrc_applied[r.site_body]-background[r.site_body],own,atol=1e-13)
            seen.append(p.copy());orig(m,d)
        with patch.object(mujoco,'mj_step',side_effect=step):
            execute_intervals(m,d,2,observe=lambda k:None,
                command=lambda k,s:Command('joint_torque',np.zeros(r.n),r.dadr),
                after_step=lambda k:None,n_substeps=4,environment=env)
        trace=env.trajectory(m,d)
        np.testing.assert_array_equal(d.xfrc_applied,background)
        np.testing.assert_allclose(trace['physics_F_ext'][:-1].sum(0)*.001,np.array([2,3,4])*.003,atol=1e-15)
        self.assertGreater(np.linalg.norm(seen[4]-seen[2]),0)
        self.assertEqual(len(trace['physics_t']),9)
        self.assertTrue(np.all(trace['physics_F_ext'][5:]==0))
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with env.physics_step(m,d,3):raise RuntimeError('injected')
        np.testing.assert_array_equal(d.xfrc_applied,background)

    def test_pic_feedback_prefix_and_discrete_impulse(self):
        case=self.case();co=pic.run(case)['traj']
        case.cfg['disturbance']={'enabled':True,'pulses':[{'t_start_s':.004,'duration_s':.003,'force_N':[10,20,0]}]}
        tr=pic.run(case)['traj']
        for key in ('Q','QD','P'):np.testing.assert_array_equal(co[key][:5],tr[key][:5])
        np.testing.assert_allclose(tr['F_ext'][:-1].sum(0)*.001,[.03,.06,0],atol=1e-15)
        self.assertTrue(np.all(tr['F_ext'][7:]==0))
        np.testing.assert_allclose(tr['physics_P'],tr['P'],rtol=0,atol=1e-14)
        self.assertGreater(np.linalg.norm(tr['P'][7]-co['P'][7]),0)

    def test_servo_uses_same_environment(self):
        case=self.case();case.cfg['disturbance']={'enabled':True,'pulses':[{'t_start_s':.002,'duration_s':.003,'force_N':[10,20,0]}]}
        tr=servo.run(case)['traj']
        np.testing.assert_allclose(tr['physics_F_ext'][:-1].sum(0)*.001,[.03,.06,0],atol=1e-15)
        self.assertTrue(np.all(tr['physics_F_ext'][5:]==0))

    def test_non_grid_pulse_is_rejected(self):
        case=self.case()
        with self.assertRaisesRegex(ValueError,'physics timestep grid'):
            TCPForceEnvironment(case.robot,[{'t_start_s':.0005,'duration_s':.003,'force_N':[1,0,0]}])

    def test_cached_gravity_and_optional_postprocessing_preserve_dynamics(self):
        from sim.robot import gravity_torque
        case=self.case();r=case.robot;cache=mujoco.MjData(r.model)
        rng=np.random.default_rng(19)
        for q in rng.uniform(r.q_min,r.q_max,size=(12,r.n)):
            np.testing.assert_array_equal(gravity_torque(r,q),gravity_torque(r,q,data=cache))
        full=pic.run(case)['traj'];light=pic.run(case,compute_tracking=False)['traj']
        for key in set(full)&set(light):np.testing.assert_array_equal(full[key],light[key])

    def test_offset_tcp_is_used(self):
        from sim.robot import Robot, ROBOTS
        r=Robot(ROBOTS['FR3'],tcp_offset=([.03,-.02,.07],np.eye(3)))
        d=mujoco.MjData(r.model);d.qpos[r.qadr]=r.q_home
        env=TCPForceEnvironment(r,[{'t_start_s':0,'duration_s':.002,'force_N':[2,3,4]}])
        with env.physics_step(r.model,d,0):
            p,_=r.fk(r.q_home)
            np.testing.assert_allclose(env.rows[-1][1],p,rtol=0,atol=1e-15)
            np.testing.assert_allclose(d.xfrc_applied[r.site_body,3:],np.cross(p-d.xipos[r.site_body],[2,3,4]))

if __name__=='__main__':unittest.main()
