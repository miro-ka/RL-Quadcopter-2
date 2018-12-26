import csv
import numpy as np
from task import Task
from agents.basic import Basic_Agent
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def run_simulation():
    # Modify the values below to give the quadcopter a different starting position.
    runtime = 5.                                     # time limit of the episode
    init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
    init_velocities = np.array([0., 0., 0.])         # initial velocities
    init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
    file_output = 'data.txt'                         # file name for saved results

    # Setup
    task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
    agent = Basic_Agent(task)
    done = False
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x : [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        while True:
            rotor_speeds = agent.act()
            _, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break

    return results


def plot_results(results, file_name):
    fig, axarr = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Simulation Results", fontsize=14)

    # Plot quadcopter position
    axarr[0, 0].plot(results['time'], results['x'], label='x')
    axarr[0, 0].plot(results['time'], results['y'], label='y')
    axarr[0, 0].plot(results['time'], results['z'], label='z')
    axarr[0, 0].legend()
    axarr[0, 0].set_title('volved position of the quadcopter')
    # Velocity
    axarr[0, 1].plot(results['time'], results['x_velocity'], label='x_hat')
    axarr[0, 1].plot(results['time'], results['y_velocity'], label='y_hat')
    axarr[0, 1].plot(results['time'], results['z_velocity'], label='z_hat')
    axarr[0, 1].legend()
    axarr[0, 1].set_title('Velocity of the quadcopter')
    #  Euler angles (the rotation of the quadcopter over the  𝑥 -,  𝑦 -, and  𝑧 -axes)
    axarr[1, 0].plot(results['time'], results['phi'], label='phi')
    axarr[1, 0].plot(results['time'], results['theta'], label='theta')
    axarr[1, 0].plot(results['time'], results['psi'], label='psi')
    axarr[1, 0].legend()
    axarr[1, 0].set_title('Euler angles (the rotation of \nthe quadcopter over x,y,x axes)')
    #
    axarr[1, 1].plot(results['time'], results['phi_velocity'], label='phi_velocity')
    axarr[1, 1].plot(results['time'], results['theta_velocity'], label='theta_velocity')
    axarr[1, 1].plot(results['time'], results['psi_velocity'], label='psi_velocity')
    axarr[1, 1].legend()
    axarr[1, 1].set_title('Velocities (rad/s) corresponding \nto each of the Euler angles.')
    #
    axarr[1, 1].plot(results['time'], results['phi_velocity'], label='phi_velocity')
    axarr[1, 1].plot(results['time'], results['theta_velocity'], label='theta_velocity')
    axarr[1, 1].plot(results['time'], results['psi_velocity'], label='psi_velocity')
    axarr[1, 1].legend()
    axarr[1, 1].set_title('Velocities (rad/s) corresponding \nto each of the Euler angles.')
    #
    axarr[2, 0].plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    axarr[2, 0].legend()
    axarr[2, 0].set_title('Agents choice of actions')

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(file_name)

    return True


def main():
    results = run_simulation()
    plot_results(results, 'plots/basic_agent_sim.png')


if __name__ == "__main__":
    main()