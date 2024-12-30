import PoseEstimator
import matplotlib.pyplot as plt
import os

estimator = PoseEstimator.PoseEstimator(debug=True)
plot = True


if(plot):
    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    #im = plt.imread("resources/field.jpg")
    #plt.imshow(im, extent=[0, 16.4846, 0, 8.1026])
    plt.plot(0, 0, "o")
    plt.ion()
    plt.show()
    plt.pause(1e-10)


while True:
    #pose_front = estimator.get_camera_pose(0, 'front')
    #pose_rear = estimator.get_camera_pose(0, 'rear')
    pose1 = estimator.get_camera_pose(0)
    #pose2 = estimator.get_camera_pose(1)

    os.system("clear")
    print("----------Fused Pose 1----------")
    print("x: %.2f" % pose1[0])
    print("y: %.2f" % pose1[1])
    print("z: %.2f" % pose1[2])
    print("r: %.2f" % pose1[3])
    print("p: %.2f" % pose1[4])
    print("y: %.2f" % pose1[5])

    '''
    print("----------Fused Pose 2----------")
    print("x: %.2f" % pose2[0])
    print("y: %.2f" % pose2[1])
    print("z: %.2f" % pose2[2])
    print("r: %.2f" % pose2[3])
    print("p: %.2f" % pose2[4])
    print("y: %.2f" % pose2[5])
    '''

    if(plot):
        plt.clf()
        plt.plot(pose1[0], pose1[1], "o")
        plt.plot(0,0,"x")
        #plt.imshow(im, extent=[0, 16.4846, 0, 8.1026])
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.show()
        plt.pause(1e-20)
