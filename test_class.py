import PoseEstimator
import matplotlib.pyplot as plt

estimator = PoseEstimator.PoseEstimator(debug=False)


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
im = plt.imread("field.jpg")
plt.imshow(im, extent=[0, 16.4846, 0, 8.1026])
plt.plot(0, 0, "o")
plt.ion()
plt.show()
plt.pause(1e-10)


while True:
    pose = estimator.get_camera_pose(0)

    print("----------Fused Pose----------")
    print("x: %.2f" % pose[0])
    print("y: %.2f" % pose[1])
    print("z: %.2f" % pose[2])
    print("r: %.2f" % pose[3])
    print("p: %.2f" % pose[4])
    print("y: %.2f" % pose[5])
    print("\n\n\n\n\n\n\n\n")

    plt.clf()
    plt.plot(pose[0], pose[1], "o")
    plt.imshow(im, extent=[0, 16.4846, 0, 8.1026])
    plt.xlim(0, 16.4846)
    plt.ylim(0, 8.1026)
    plt.show()
    plt.pause(1e-10)
