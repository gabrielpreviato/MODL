import numpy as np


class Classes(object):
    def __init__(self, class_enum):
        self.class_enum = class_enum

        self.is_nothing = False
        self.color = np.array((0, 0, 255))

    @staticmethod
    def generate_class(class_enum):
        if class_enum == 0:
            return Robot()
        elif class_enum == 1:
            return Ball()
        elif class_enum == 2:
            return Goal()
        elif class_enum == 3:
            return RobotTeam()
        elif class_enum == 4:
            return RobotOpponent()
        else:
            return Nothing()

    @staticmethod
    def str_to_class_enum(string):
        if string == 'robot':
            return 0
        elif string == 'ball':
            return 1
        elif string == 'goal':
            return 2
        elif string == 'robot_team':
            return 3
        elif string == 'robot_opponent':
            return 4
        else:
            return -1


class Nothing(Classes):
    def __init__(self):
        super(Nothing, self).__init__(0)

        self.is_nothing = True

        self.color = np.array((0, 0, 0))


class Robot(Classes):
    def __init__(self):
        super(Robot, self).__init__(0)

        self.color = np.array((255, 0, 0))


class RobotTeam(Classes):
    def __init__(self):
        super(RobotTeam, self).__init__(0)

        self.color = np.array((0, 255, 0))


class RobotOpponent(Classes):
    def __init__(self):
        super(RobotOpponent, self).__init__(0)

        self.color = np.array((255, 255, 0))


class Ball(Classes):
    def __init__(self):
        super(Ball, self).__init__(1)

        self.color = np.array((255, 0, 255))


class Goal(Classes):
    def __init__(self):
        super(Goal, self).__init__(2)

        self.color = np.array((0, 255, 255))
