import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_results(results, file_name):
    fig, axarr = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Simulation Results", fontsize=14, y=1.03)

    # Plot quadcopter position
    axarr[0, 0].plot(results['time'], results['x'], label='x')
    axarr[0, 0].plot(results['time'], results['y'], label='y')
    axarr[0, 0].plot(results['time'], results['z'], label='z')
    axarr[0, 0].legend()
    axarr[0, 0].set_title('Evolved position of the quadcopter')
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
    axarr[1, 0].set_title('Euler angles (rotation over x,y,x axes)')
    #
    axarr[1, 1].plot(results['time'], results['phi_velocity'], label='phi_velocity')
    axarr[1, 1].plot(results['time'], results['theta_velocity'], label='theta_velocity')
    axarr[1, 1].plot(results['time'], results['psi_velocity'], label='psi_velocity')
    axarr[1, 1].legend()
    axarr[1, 1].set_title('Velocities (rad/s) corresponding to each of the Euler angles.')
    #
    axarr[1, 1].plot(results['time'], results['phi_velocity'], label='phi_velocity')
    axarr[1, 1].plot(results['time'], results['theta_velocity'], label='theta_velocity')
    axarr[1, 1].plot(results['time'], results['psi_velocity'], label='psi_velocity')
    axarr[1, 1].legend()
    axarr[1, 1].set_title('Velocities (rad/s) corresponding to each of the Euler angles.')
    #
    axarr[2, 0].plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
    axarr[2, 0].plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
    axarr[2, 0].legend()
    axarr[2, 0].set_title('Agents choice of actions')

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    fig.tight_layout()
    # fig.subplots_adjust(top=1.2)
    plt.savefig(file_name)

    return True
