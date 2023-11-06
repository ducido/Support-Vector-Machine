import pygame
import color
import threading
from sklearn.cluster import KMeans
import numpy as np
from hinge_loss import *
import time
from sklearn.svm import SVC
from hard_margin import Hard_margin
from sklearn.linear_model import LogisticRegression

def draw_line(w, b):
    print(np.round(w, 3), np.round(b, 3))
    a = []
    for x in range(1094, 15, -1):
        y = (w[0][0]*x + b)/(-w[1][0])
        if y <= 16 or y >= 564:
            continue
        a.append((x,y))
    return a

draw = False
def time_draw():
    global draw
    time.sleep(2)
    draw = False

def predict(X, w, b):
    prediction = np.dot(X, w) + b # w.x + b
    return np.sign(prediction)

pygame.init()
screen = pygame.display.set_mode((1110, 700))
caption = pygame.display.set_caption("SVM Algorithm")

mouse_x, mouse_y = 0,0
def set_up():
    global mouse_x, mouse_y
    #draw panel
    pygame.draw.rect(screen, color.WHITE, (15,15,1080,550))
    pygame.draw.rect(screen, color.BLACK, (16,16,1078,548))
    #draw class 1 2
    pygame.draw.rect(screen, color.WHITE, (100, 596, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (101,597,78,28))
    screen.blit(text_class1, (105,599.5))
    pygame.draw.rect(screen, color.WHITE, (100, 650, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (101, 651,78,28))
    screen.blit(text_class2, (105,653.5))
    #draw train
    pygame.draw.rect(screen, color.WHITE, (223, 596, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (224,597,78,28))
    screen.blit(text_train, (238,599.5))
    #draw predict
    pygame.draw.rect(screen, color.WHITE, (223, 650, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (224,651,78,28))
    screen.blit(text_predict, (226,653.5))
    #draw hard margin dual
    pygame.draw.rect(screen, color.WHITE, (340, 650, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (341,651,78,28))
    screen.blit(text_hard, (348,653.5))
    #draw logistic
    pygame.draw.rect(screen, color.WHITE, (457, 650, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (458,651,78,28))
    screen.blit(text_logistic, (465,653.5))
    #draw train minibatch
    pygame.draw.rect(screen, color.WHITE, (584, 650, 80, 30))
    pygame.draw.rect(screen, color.GRAY, (585,651,78,28))
    screen.blit(text_train_mnb, (590,653.5))
    #draw random
    pygame.draw.rect(screen, color.WHITE, (339, 596, 110, 30))
    pygame.draw.rect(screen, color.GRAY, (340,597,108,28))
    screen.blit(text_random, (356.5, 599))
    #draw algorithm
    pygame.draw.rect(screen, color.WHITE, (485, 596, 185, 30))
    pygame.draw.rect(screen, color.PINK, (486,597,183,28))
    screen.blit(text_al, (506.5, 599))
    #draw reset
    pygame.draw.rect(screen, color.WHITE, (707, 596, 100, 30))
    pygame.draw.rect(screen, color.GRAY, (708,597,98,28))
    screen.blit(text_reset, (726,599))
    #draw number of points
    pygame.draw.rect(screen, color.RED, (674, 670, 400, 30))
    pygame.draw.rect(screen, color.BLACK, (675, 671, 223, 28))
    text_nop = font_small_2.render('NUMBER OF POINTS : ' + str(len(points)), True, color.WHITE)
    screen.blit(text_nop, (700,675))
    #draw errors
    pygame.draw.rect(screen, color.BLACK, (899, 671, 174, 28))
    text_errors = font_small_2.render('ERRORS : ' + str(round(error, 3)), True, color.WHITE)
    screen.blit(text_errors, (940,675))
    #draw mouse position
    mouse_x, mouse_y = pygame.mouse.get_pos()
    if 16 <= mouse_x <= 1094 and 16 <= mouse_y <= 564:
        text_mouse = font_small_2.render(f"({mouse_x - 16},{mouse_y - 16})", True, color.WHITE)
        screen.blit(text_mouse, (mouse_x + 17, mouse_y + 17))

running = True
clock = pygame.time.Clock()

# font
font = pygame.font.SysFont('sans', 24)
font_big_1 = pygame.font.SysFont('sans', 30)
font_small_1 = pygame.font.SysFont('sans', 20)
font_small_2 = pygame.font.SysFont('sans', 18)
# text
text_minus = font_big_1.render('-', True, color.WHITE)
text_plus = font_big_1.render('+', True, color.WHITE)
text_train = font_small_1.render('TRAIN', True, color.WHITE)
text_train_mnb = font_small_1.render('TRAIN B', True, color.WHITE)
text_random = font_small_1.render('RANDOM', True, color.WHITE)
text_al = font_small_1.render('USE ALGORITHM', True, color.WHITE)
text_reset = font_small_1.render('RESET', True, color.WHITE)
text_class1 = font_small_1.render('CLASS 1', True, color.WHITE)
text_class2 = font_small_1.render('CLASS 2', True, color.WHITE)
text_predict = font_small_1.render('PREDICT', True, color.WHITE)
text_hard = font_small_1.render('HARD', True, color.WHITE)
text_logistic = font_small_1.render('LOGISTIC', True, color.WHITE)

iter = 0
error = 0
points = []
valid_list = []

w0 = None
class1 = False
class2 = False
svm_batch = SVM_HingeLoss_MiniBatch()
svm = SVM_HingeLoss()
hard_dual = Hard_margin()
logistic_model = LogisticRegression()

while running:
    clock.tick(60)
    screen.fill(color.BACKGROUND)

    set_up()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # active TRAIN
            if 224 < mouse_x < 302 and 597 < mouse_y < 625:
                if w0 is not None and points:
                    w_list = []
                    train_p = [i for i in points if i[-1] != 0]
                    X, y = process(train_p)
                    w_list, b_list, error = svm.fit(X,y, w0, b0)
                    draw = True
                    for i in range(len(w_list)):
                        valid_0 = draw_line(w_list[i], b_list[i])
                        if valid_0:
                            x1, y1, x2, y2 = valid_0[0][0], valid_0[0][1], valid_0[-1][0], valid_0[-1][1]
                            valid_list.append([x1, y1, x2, y2])

                    threading.Thread(target= time_draw).start()
                    w = w_list[-1]
                    b = b_list[-1]
                    w0 = w
                    b0 = b
                    valid = draw_line(w, b)

            # active RANDOM
            if 340 < mouse_x < 448 and 597 < mouse_y < 625:
                valid_list = []
                draw = False
                w0 = np.random.uniform(-1,1,(2, 1))
                b0 = np.random.uniform(-1,1) * 200
                valid = draw_line(w0, b0)

            # active AL
            if 486 < mouse_x < 669 and 597 < mouse_y < 625:
                draw = False
                train_p = [i for i in points if i[-1] != 0]
                X, y = process(train_p)

                clf = SVC(kernel = 'linear', C = 100)
                clf.fit(X, y.T) 

                w = clf.coef_.reshape(-1, 1)
                b = clf.intercept_[0]
                valid = draw_line(w, b)
                pass

            # active hard dual
            if 341 < mouse_x < 419 and 651 < mouse_y < 679:
                draw = False
                train_p = [i for i in points if i[-1] != 0]
                print(train_p)
                X, y = process(train_p)
                w, b = hard_dual.fit(X, y)
                valid = draw_line(w, b)
                 
                pass

            # active logistic regression
            if 458 < mouse_x < 536 and 651 < mouse_y < 679:
                train_p = [i for i in points if i[-1] != 0]
                X, y = process(train_p)

                logistic_model.fit(X, y.T)

                w = logistic_model.coef_
                b = logistic_model.intercept_[0]
                valid = draw_line(w.T, b)
                pass

            # active train batch
            if 585 < mouse_x < 663 and 651 < mouse_y < 679:
                if w0 is not None and points:
                    w_list = []
                    train_p = [i for i in points if i[-1] != 0]
                    X, y = process(train_p)
                    w_list, b_list, error = svm_batch.fit(X,y, w0, b0)
                    draw = True
                    for i in range(len(w_list)):
                        valid_0 = draw_line(w_list[i], b_list[i])
                        if valid_0:
                            x1, y1, x2, y2 = valid_0[0][0], valid_0[0][1], valid_0[-1][0], valid_0[-1][1]
                            valid_list.append([x1, y1, x2, y2])

                    threading.Thread(target= time_draw).start()
                    w = w_list[-1]
                    b = b_list[-1]
                    w0 = w
                    b0 = b
                    valid = draw_line(w, b)

            # active predict
            if 224 < mouse_x < 302 and 651 < mouse_y < 679:
                pre_p = [i for i in points if i[-1] == 0]
                if pre_p:
                    X, y = process(pre_p)
                    pred = predict(X, w, b)
                    p = np.concatenate([X, pred], axis= 1)
                    points += list(p)
                pass
            
            # active RESET
            if 708 < mouse_x < 806 and 597 < mouse_y < 625:
                iter = 0
                error = 0
                points = []
                valid = []
                valid_list = []
                w0 = None
                class1 = False
                class2 = False
                draw = False

                pass

            # active class  (101,597,78,28))
            if 101 < mouse_x < 179 and 597 < mouse_y < 625:
                class1 = True
                class2 = False
            elif 101 < mouse_x < 179 and 651 < mouse_y < 679:
                class2 = True
                class1 = False
            elif not (16 <= mouse_x <= 1094 and 16 <= mouse_y <= 564):
                class1 = False
                class2 = False

            # create points
            if 16 <= mouse_x <= 1094 and 16 <= mouse_y <= 564:
                if class1:
                    point = [mouse_x - 16, mouse_y - 16]
                    for _ in range(10): 
                        new = point + np.random.uniform(-2,2,(2))*20
                        new = list(new)
                        new.append(1)
                        points.append(new)
                    point.append(1)

                elif class2:
                    point = [mouse_x - 16, mouse_y - 16]
                    for _ in range(10): 
                        new = point + np.random.uniform(-2,2,(2))*20
                        new = list(new)
                        new.append(-1)
                        points.append(new)
                    point.append(-1)
                else:
                    point = [mouse_x - 16, mouse_y - 16, 0]
                points.append(point)

    # draw line
    try:
        pygame.draw.line(screen, color.PINK, (valid[0][0], valid[0][1]), (valid[-1][0], valid[-1][1]))
        if draw:      
            i = 0
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))
            i += 10
            pygame.draw.line(screen, color.PINK, (valid_list[i][0], valid_list[i][1]), (valid_list[i][2], valid_list[i][3]))

    except: pass

    #draw points
    for i in range(len(points)):
        if points[i][-1] == 1:
            pygame.draw.circle(screen, color.MAROON, (points[i][0] + 16, points[i][1] + 16), 5)
        elif points[i][-1] == -1:
            pygame.draw.circle(screen, color.GREEN, (points[i][0] + 16, points[i][1] + 16), 5)
        else:
            pygame.draw.circle(screen, color.WHITE, (points[i][0] + 16, points[i][1] + 16), 5)


    # draw class
    if class1:
        pygame.draw.rect(screen, color.SKY, (101,597,78,28))
        screen.blit(text_class1, (105,599.5))
    if class2:
        pygame.draw.rect(screen, color.SKY, (101, 651,78,28))
        screen.blit(text_class2, (105,653.5))

    # draw error
    pygame.display.flip()
pygame.quit()