# reference website: https://www.pygame.org/docs/ref/joystick.html
import pygame
from termcolor import colored
import numpy as np
import _thread
import threading
import time
import sys

class Pygame_text_board:

    def __init__(self, windows_size=[500,700]):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        pygame.init()
        self.screen = pygame.display.set_mode(windows_size)
        pygame.display.set_caption("text_board")
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def print(self, textString):
        text_line = textString.split('\n')
        for str_line in text_line:
            textBitmap = self.font.render(str_line, True, self.BLACK)
            self.screen.blit(textBitmap, [self.x, self.y])
            self.y += self.line_height

    def reset(self):
        self.screen.fill(self.WHITE)
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

    def show(self):
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.display.flip()

class Joystick_controller:
    def __init__(self):
        self.m_joystick_value = []
        self.m_joystick=[]
        self.m_is_init = False
        self.m_thr = 0
        self.m_yaw = 0
        self.m_roll = 0
        self.m_pitch = 0

    def refine_channels(self):
        self.m_thr = -self.m_joystick_value[1]
        self.m_yaw =  self.m_joystick_value[0]
        self.m_pitch =  self.m_joystick_value[3]
        self.m_roll =  self.m_joystick_value[4]

    def init(self):
        pygame.init()
        joystick_count = pygame.joystick.get_count()

        if joystick_count !=0:
            self.m_joystick = pygame.joystick.Joystick(0)
            self.m_joystick.init()
            name = self.m_joystick.get_name()
            axes = self.m_joystick.get_numaxes()
            print("Joystick count = ", joystick_count)
            print("name = %s, axis_number = %d" % (name, axes))
            pygame.display.set_mode([1, 1])
            pygame.joystick.init()
            try:
                # self.refresh_service();
                mthread = threading.Thread(target=self.refresh_service)
                mthread.start();
                self.m_is_init = True
                print(colored("Joystick_controller","blue") ,colored("run as a service.","red"))    #color print
                # _thread.start_new_thread(self.refresh_service);
            except Exception as e:
                print(e)

    def refresh_service(self):
        while (1):
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.m_joystick.init()
            axes = self.m_joystick.get_numaxes()
            self.m_joystick_value = np.zeros(axes)
            for i in range(axes):
                axes_val = self.m_joystick.get_axis(i)
                axis = self.m_joystick.get_axis(i)
                self.m_joystick_value[i] = round(axis, 3)
                # self.m_joystick_value[i] = round(axis,2)
                # print(colored( "axis %d, val = %.3f" % ((i), axis), "yellow"))
            # print("Service: " , self.m_joystick_value)
            self.refine_channels();
            time.sleep(0.02)

    def display_current_val(self):
        # np.set_printoptions(precision=2)
        # print(self.m_joystick_value)
        ss = " Joystick_thr = %.5f\n Joystick_yaw = %.5f\n Joystick_pitch = %.5f\n Joystick_roll = %.5f\n"%(self.m_thr, self.m_yaw, self.m_pitch, self.m_roll )
        return ss

# Test joystick class
if __name__ == "__main__":
    joystick_controller = Joystick_controller();
    joystick_controller.init();
    # joystick_controller.refresh_service();
    while joystick_controller.m_is_init:
        print("Main: ", joystick_controller.m_joystick_value)