# self-driving car in a simulator

Behavioral cloning in Udacity's car simulator: a CNN looks at the front camera and outputs a steering angle. It's Nvidia's 2016 end-to-end network (about 250k weights) in PyTorch, trained on my RTX 3070.

The usual way to do this is to drive a few laps yourself and have the network copy you. I didn't want to drive, so I wrote a dumb hand-coded driver to do the laps and had the network copy that instead.

## the teacher

`car/road_detect.py` finds the road by sampling the color right in front of the car and keeping the patch of similar color that's connected to it. The yellow edge lines are treated as walls. Then it steers toward the middle of that patch (red dot).

![road mask](media/road_mask.png)

The first version just looked for gray pixels, and since the lake is kind of gray it drove straight into the bridge wall. Sampling the road color fixed that, but then it would drift onto the concrete shoulder, decide the shoulder was the road, and follow it off the track. Cutting the mask at the yellow lines fixed that one and now it does full laps at 9 mph.

To get recovery examples, the teacher swerves on purpose for about half a second every few seconds, but it always saves the steering it *would* have used. So the data has lots of "you're drifting, steer back" moments without me having to record them by hand.

I recorded 6000 frames at 10 per second (about 2 laps) and cut one stretch where the teacher ran wide onto the grass.

## result

Training took about 4 minutes on the GPU. Validation loss went from 0.030 to 0.012 and stayed right next to training loss, so no real overfitting.

![loss](media/loss.png)

Then the network drove track 1 by itself at 12 mph, which is faster than the teacher it learned from, and did full laps without leaving the road.

## run it (Windows)

1. `setup.bat` installs everything and downloads the simulator
2. open `sim\beta_simulator_windows\beta_simulator.exe`, pick the left track
3. `teacher.bat`, then Autonomous Mode in the sim. It records into `data\teacher`
4. `train_car.bat` writes `model.pt` (the one I trained is already in the repo if you want to skip to driving)
5. `drive_car.bat`, then Autonomous Mode again

The simulator speaks an old socket.io version, so `python-socketio==4.6.1` and `python-engineio==3.13.2` are pinned. Don't upgrade those or it won't connect.

You can also record yourself in Training Mode into any folder under `data\` and `train.py` will pick it up, side cameras included.
