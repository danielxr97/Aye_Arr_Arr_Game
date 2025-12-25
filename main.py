import pygame
import numpy as np
import random
#import csv
#import os

# Base sizes
base_width = 800
base_height = 600
WIDTH = base_width
HEIGHT = base_height

current_screen = "Menu"
settings_open = False
volume = 0.5
muted = False
selected_weapon = None
selected_level = None
music_started = False

# Pirate shots settings
pirate_cannon_bullets = []
pirate_arc_points = []
pirate_cannon_x = 0
pirate_cannon_y = HEIGHT // 2
# Settings for the Parrot Game
parrot_bullets = [] # Active shots for parrot
parrot_shoot = True # This tracks the state of the Spacebar when shooting


# Settings for the Cannon Game
angle_deg = 0 # Cannon actual angle (degrees)
deg_to_show = 25 # Cannon base angle (degrees)
power = 0
cannon_bullets = [] # We need a list to track the active shots
cannon_shoot = True # This tracks the state of the Spacebar when shooting
cannon_offset_x = -60
cannon_offset_y = -30

# Settings for the Blunderbuss Game
blunderbuss_aim_angle_deg = 0
blunderbuss_angle_deg = 0
blunderbuss_bullets = []
blunderbuss_shoot = True
blunderbuss_offset_x = -73
blunderbuss_offset_y = -6

# Settings for Machine Learning
difficulty_sd = 0.05
capital_g = 0.0 # Accumulated Gradient
memory_file = "pirate_across-game_memory_data/pirate_memory.csv"
alpha = 0.3 # Damping factor. The learning rate
alpha_min, alpha_max = 0.05, 1.0
"""
Alpha helps us regulate the machine learning curve because sometimes the slope can be 
too steep and then it will overshoot the answer by a significant number.
"""

# AdaGrad state for alpha
alpha_g = 0.0

# Adam state for alpha
alpha_m = 0.0 # First moment - momentum and velocity
alpha_v = 0.0 # Second moment - variance and volatility
alpha_t = 0 # time step - bias correction counter


pirate_r_list = []
player_r_list = []
landing_x = 0
player_r_guess = 0 # The player's IRR guess
shot_message = ""
booty = random.randint(1, 10) # The CFs for the NPV
blood = 0 # This is the initial cost for a project, for example

plunder_text = ""
plunder_box_active = False
plunders = 1

easy_box_active = False
medium_box_active = False
hard_box_active = False

game_over = False
current_turn = "Player"
rounds_left = plunders
pirate_r_guess = None
midpoint = [0, 1.0]
player_message_timer = 0.0
pending_pirate_turn = False
pirate_message_timer = 0.0 # How long to keep the Pirate's message
pirate_has_acted = False # An indicator of whether the Pirate made his turn
pending_player_turn = False
player_turn_message_timer = 0.0
last_player_message = ""

# Saving the pixel text font
pygame.font.init()
font_18 = pygame.font.Font("fonts/pixel_reg.ttf", 18)
font_22 = pygame.font.Font("fonts/pixel_reg.ttf", 22)
font_25 = pygame.font.Font("fonts/pixel_reg.ttf", 25)
font_26 = pygame.font.Font("fonts/pixel_reg.ttf", 26)
font_30 = pygame.font.Font("fonts/pixel_reg.ttf", 30)
font_32 = pygame.font.Font("fonts/pixel_reg.ttf", 32)

# Creating a function for multiline text
def draw_multiline_text(screen, text, font, color, pos, line_spacing=0):
    x, y = pos
    for line in text.split("\n"):
        surface = font.render(line, True, color) # The True makes the edges of the text smoother
        screen.blit(surface, (x, y))
        y += surface.get_height() + line_spacing

# Screen Setup
def main():
    global screen, keys
    pygame.init()
    pygame.mixer.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Aye Arr Arr (IRR) Game") # Setting the Windows title
    clock = pygame.time.Clock()

    init_assets()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                on_mouse_down(event.pos)
            elif event.type == pygame.KEYDOWN:
                on_key_down(event.key)

        update()
        draw()
        pygame.display.flip()

    pygame.quit()


# Creating the blueprint for the objects
class ActorLike:
    # Initializing the image / object
    def __init__(self, image_name, images_dir="images"):
        # image_name like "btn_play" -> images/btn_play.png
        self.image_name = image_name # self is the object itself
        path = f"{images_dir}/{image_name}.png"
        self.image = pygame.image.load(path).convert_alpha()
        self.rect = self.image.get_rect()
        self._angle = 0

    # The property returns the value for the variable
    @property
    def pos(self):
        return self.rect.center
    
    # The setter sets / updates the positions or new value for that variable, so it doesn't need to return anything
    @pos.setter
    def pos(self, value):
        self.rect.center = value

    @property
    def x(self):
        return self.rect.centerx
    
    @x.setter
    def x(self, v):
        self.rect.centerx = v

    @property
    def y(self):
        return self.rect.centery
    
    @y.setter
    def y(self, v):
        self.rect.centery = v

    @property
    def left(self):
        return self.rect.left
    
    @left.setter
    def left(self, v):
        self.rect.left = v

    @property
    def right(self):
        return self.rect.right
    
    @right.setter
    def right(self, v):
        self.rect.right = v

    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, a):
        self._angle = a

    @property
    def top(self):
        return self.rect.top
    
    @top.setter
    def top(self, v):
        self.rect.top = v

    @property
    def bottom(self):
        return self.rect.bottom
    
    @bottom.setter
    def bottom(self, v):
        self.rect.bottom = v

    @property
    def width(self):
        return self.rect.width
    
    @property
    def height(self):
        return self.rect.height

    def collidepoint(self, pos):
        return self.rect.collidepoint(pos)
    
    def draw(self, screen):
        if self._angle != 0:
            rotated = pygame.transform.rotate(self.image, self._angle)
            r = rotated.get_rect(center=self.rect.center)
            screen.blit(rotated, r)
        else:
            screen.blit(self.image, self.rect)


def init_assets():
    global btn_play, btn_settings, btn_vol_up, btn_vol_down, btn_vol_mute
    global btn_instructions, btn_back, btn_start
    global btn_parrot, btn_cannon, btn_blunderbuss, btn_easy, btn_medium, btn_hard
    global plunders_box_img
    global plunders_box, easy_box, medium_box, hard_box
    global player_ship, weapon_parrot, weapon_cannon, weapon_blunderbuss
    global parrot_bullet, cannon_bullet, blunderbuss_bullet, pirate_bullet_actor
    
    # Button Actors
    btn_play = ActorLike("btn_play")
    btn_settings = ActorLike("btn_settings")
    btn_vol_up = ActorLike("btn_volume_up")
    btn_vol_down = ActorLike("btn_volume_down")
    btn_vol_mute = ActorLike("btn_volume_mute")
    btn_instructions = ActorLike("btn_instructions")
    btn_back = ActorLike("btn_back")
    btn_start = ActorLike("btn_play")

    # Selection Screen Actors
    btn_parrot = ActorLike("btn_parrot")
    btn_cannon = ActorLike("btn_cannon")
    btn_blunderbuss = ActorLike("btn_blunderbuss")
    btn_easy = ActorLike("btn_easy")
    btn_medium = ActorLike("btn_medium")
    btn_hard = ActorLike("btn_hard")
    plunders_box_img = ActorLike("plunders_box")
    plunders_box = pygame.Rect(200, 275, 100, 40)
    easy_box = pygame.Rect(375, 132, 150, 86)
    medium_box = pygame.Rect(375, 282, 150, 86)
    hard_box = pygame.Rect(375, 432, 150, 86)

    # In-Game Actors
    player_ship = ActorLike("player_ship")
    weapon_parrot = ActorLike("weapon_parrot")
    weapon_cannon = ActorLike("weapon_cannon")
    weapon_blunderbuss = ActorLike("weapon_blunderbuss")
    cannon_bullet = ActorLike("weapon_cannon_bullet")
    parrot_bullet = ActorLike("weapon_parrot_bullet")
    blunderbuss_bullet = ActorLike("weapon_blunderbuss_bullet")
    pirate_bullet_actor = ActorLike("weapon_cannon_bullet")


for i in range(100):
    pirate_r_list.append(round(random.uniform(0.2, 0.7), 2)) # Drawing 5 random locations (r values) for the pirate

pirate_current_r = round(random.choice(pirate_r_list), 2) # Selecting a location for the pirate at random

for i in range(100):
    player_r_list.append(round(random.uniform(0.2, 0.7), 2)) # Drawing 5 random locations (r values) for the pirate

player_current_r = random.choice(player_r_list) # Selecting a location for the pirate at random

pirate_r_guess = round(random.uniform(0.2, 0.7), 2)

# Central difference derivative approximation of f at r
def numerical_derivative(f, r, h = 1e-5):
    r_forward = min(1.0, r + h)
    r_backward = max(1e-8, r - h)
    derivative = (f(r_forward) - f(r_backward)) / (r_forward - r_backward)

    return derivative

def npv_fn(r, booty, t, blood):
    npv = booty / r * (1 - (1 + r)**(-t)) - blood # Since it's an annuity
    return npv

def npv_zero(my_r, opponent_r, t, booty, alpha_in=None):
    global blood, capital_g, alpha

    npv = 0

    # Using the learned alpha by default
    if alpha_in is None:
        alpha_in = alpha

    if my_r is None:
        my_r = 1e-6 # Avoids division by None
    if opponent_r is None:
        opponent_r = 1e-6

    if my_r == 0:
        my_r = 1e-6 # Avoids division by None
    if opponent_r == 0:
        opponent_r = 1e-6

    blood_local = booty / opponent_r * (1 - (1 + opponent_r)**(-t)) - npv # Here we're making sure that the Pirate's r location will make the NPV = 0 by fitting the blood to the right value
    

    f = lambda r: npv_fn(r, booty, t, blood_local) # When we call f the input will be r and the output will be the NPV

    npv = f(my_r)

    try: # Trying to compute the numerical derivative
        npv_derivative = numerical_derivative(f, my_r) # If it works we do this
    except ZeroDivisionError: # If we're dividing by 0, we do this:
        npv_derivative = 0.0

    if abs(npv_derivative) > 1e-6:
        newton_step = -npv / npv_derivative
        x = my_r + alpha_in * newton_step # This is the x value for the Linear Approximation equation, which will correspond to the Pirate's r guess
    else: # In case that npv_derivative = 0, or technically, < 1e-6, we fall back to the bisection method
        newton_step = 0.0
        x = (my_r + opponent_r) / 2

    x = max(0.01, min(0.99, x))

    return blood_local, x, npv, npv_derivative, newton_step

# AdaGrad Calculation
def adagrad(r, npv, npv_derivative):
    global capital_g

    eta = 0.1 # Learning rate for the gradient - Same as alpha but for AdaGrad
    epsilon = 1e-8
    gradient = 0 # Gradient

    gradient = npv * npv_derivative

    capital_g += gradient**2

    # Gradient descent
    r_guess = r - eta * gradient / (capital_g + epsilon)**0.5
    r_guess = max(0.01, min(0.99, r_guess))

    return r_guess, gradient, capital_g

def adagrad_alpha_step(alpha, grad_alpha):
    global alpha_g

    eta = 0.15
    epsilon = 1e-8
    alpha_g += grad_alpha**2
    
    alpha = alpha - eta * grad_alpha / ((alpha_g + epsilon)**0.5)
    alpha = max(alpha_min, min(alpha_max, alpha))

    return alpha

# Adam Optimization
def adam_alpha_step(alpha, grad_alpha):
    global alpha_m, alpha_v, alpha_t

    eta = 0.08
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    alpha_t += 1
    alpha_m = beta_1 * alpha_m + (1 - beta_1) * grad_alpha
    alpha_v = beta_2 * alpha_v + (1 - beta_2) * grad_alpha**2

    m_hat = alpha_m / (1 - beta_1**alpha_t)
    v_hat = alpha_v / (1 - beta_2**alpha_t)

    alpha = alpha - eta * m_hat / ((v_hat**0.5) + epsilon)
    alpha = max(alpha_min, min(alpha_max, alpha))

    return alpha


def loss_from_r(r, booty, t, blood_local):
    f = lambda x: npv_fn(x, booty, t, blood_local)
    val = f(r)
    loss = 0.5 * val**2

    return loss

def grad_alpha_derivative(r, newton_step, alpha, booty, t, blood_local, epsilon=1e-3): # newton_step = - NPV(r) / NPV'(r)
    a_plus = max(alpha_min, min(alpha_max, alpha + epsilon))
    a_minus = max(alpha_min, min(alpha_max, alpha - epsilon))

    r_plus = max(0.01, min(0.99, r + a_plus * newton_step))
    r_minus = max(0.01, min(0.99, r + a_minus * newton_step))

    loss_plus = loss_from_r(r_plus, booty, t, blood_local)
    loss_minus = loss_from_r(r_minus, booty, t, blood_local)

    derivative = (loss_plus - loss_minus) / (a_plus - a_minus) # This is the numerical derivative of L(α​)

    return derivative

"""
# Across-Game Optimization Memory
def across_game_optimization(booty, blood, plunders, pirate_r, player_location, npv, error, loss, gradient):
    os.makedirs(os.path.dirname(memory_file), exist_ok=True)
    file_exists = os.path.exists(memory_file)

    try:
        with open(memory_file, "a", newline="") as f:
            fieldnames = [
                "Booty", "Blood", "Plunders",
                "Player's Current r", "Pirate's r Guess",
                "NPV","Error", "Loss Function", "gradient"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "Booty": booty,
                "Blood": blood,
                "Plunders": plunders,
                "Player's Current r": player_location,
                "Pirate's r Guess": pirate_r,
                "NPV": npv,
                "Error": error, # pirate_r_guess - player_current_r
                "Loss Function": loss, # L(r) = ½ * NPV(r)^2
                "gradient": gradient # Derivative of the Loss Function: npv * npv_derivative
            })
    
    except OSError:
        pass
"""

def pirate_hard_mode(pirate_r_guess, player_current_r, plunders, booty, method="adam"):
    """
    Hard Mode:
    1) Compute Newton step components
    2) Compute d/dalpha of loss after taking (r + alpha * newton_step)
    3) Update alpha via Adam or AdaGrad
    4) Take the Newton step with the updated alpha
    """
    global alpha

    blood_local, x_temp, npv_val, npv_der, newton_step = npv_zero(
        pirate_r_guess, player_current_r, plunders, booty
    )

    # Only optimize alpha when Newton is valid: derivative is bigger than zero - no division by zero
    if abs(npv_der) > 1e-6:
        g_alpha = grad_alpha_derivative(
            r = pirate_r_guess,
            newton_step = newton_step,
            alpha = alpha,
            booty = booty,
            t = plunders,
            blood_local = blood_local
        )

        if method == "adagrad":
            alpha = adagrad_alpha_step(alpha, g_alpha)
        else:
            alpha = adam_alpha_step(alpha, g_alpha)

    # Updating alpha
    blood_local, x, npv_val, npv_der, newton_step = npv_zero(
        pirate_r_guess, player_current_r, plunders, booty, alpha_in=alpha
    )

    pirate_r_guess = max(0.01, min(0.99, x)) # We're doing this to make sure that Newton–Raphson Method doesn't produce value outside of our range

    return pirate_r_guess, blood_local, npv_val, npv_der


# Using angle_deg input but we need to reset is to None
def barrel_tip(offset_x, offset_y, weapon_selected, angle, angle_deg = None):
    # If no angle is given, use the base blunderbuss angle
    if angle_deg is None:
        angle_deg = angle
    
    rad = np.deg2rad(angle_deg)

    # Adjusting the axes to their rotating by taking the derivative of x' and y' to adjust.
    rotated_x = offset_x * np.cos(rad) - offset_y * np.sin(rad)
    rotated_y = offset_x * np.sin(rad) + offset_y * np.cos(rad)

    start_x = weapon_selected.x + rotated_x
    start_y = weapon_selected.y + rotated_y

    return start_x, start_y

# Creating the arc line for the Cannon's aim
def draw_cannon_arc():
    g = 9.81 # Gravity
    t = 0 # Time step, which is also the accuracy of the line
    v = power # Velocity
    deg_to_show_rad = np.deg2rad(deg_to_show)

    start_x, start_y = barrel_tip(cannon_offset_x, cannon_offset_y, weapon_cannon, deg_to_show, angle_deg)

    positions = []

    while True:
        x = v * np.cos(deg_to_show_rad) * t # # Using the physics equation y(t) = v * cos(θ) * t to find the x value of the point
        y = v * np.sin(deg_to_show_rad) * t - 0.5 * g * t**2 # Using the physics equation y(t) = v * sin(θ) * t -0.5 * g * t^2 to find the y value of the point

        screen_x = start_x - x # This will give us the distance from the current point to the next on x-axis
        screen_y = start_y - y # This will give us the distance from the current point to the next on y-axis

        if (screen_x < 0 or screen_y > HEIGHT):
            break

        positions.append((screen_x, screen_y))
        t += 0.2 # Time step, which is also the accurancy of the line

    for i in range(len(positions) - 1):
        pygame.draw.line(screen, (255,255,0), positions[i], positions[i + 1], 2) # For the colour yellow


# Drawing the arc for the Blunderbuss
def draw_blunderbuss_arc(shaken_angle_deg = 0):
    g = 9.81 # Gravity
    t = 0 # Time step, which is also the accuracy of the line
    v = power # Velocity

    # Use the shaken angle for the arc
    angle_deg = blunderbuss_angle_deg + shaken_angle_deg
    angle_rad = np.deg2rad(angle_deg)
    
    start_x, start_y = barrel_tip(blunderbuss_offset_x, blunderbuss_offset_y, weapon_blunderbuss, blunderbuss_angle_deg, angle_deg)

    positions = []

    while True:
        x = v * np.cos(angle_rad) * t # # Using the physics equation y(t) = v * cos(θ) * t to find the x value of the point
        y = v * np.sin(angle_rad) * t - 0.5 * g * t**2 # Using the physics equation y(t) = v * sin(θ) * t -0.5 * g * t^2 to find the y value of the point

        screen_x = start_x - x # This will give us the distance from the current point to the next on x-axis
        screen_y = start_y - y # This will give us the distance from the current point to the next on y-axis

        if (screen_x < 0 or screen_y > HEIGHT):
            break

        positions.append((screen_x, screen_y))
        t += 0.2 # Time step, which is also the accurancy of the line

    for i in range(len(positions) - 1):
        pygame.draw.line(screen, (255,255,0), positions[i], positions[i + 1], 2)

# Finding the pirate's shooting angle, θ, for the physics equation
def pirate_shooting_angle(start, target, v, g=9.81):
    x0, y0 = start
    x1, y1 = target

    dx = x1 - x0
    dy = y1 - y0
    
    if dx < 1e-6:
        return None # When the shot is vertical
    
    k = g * dx**2 / (2 * v**2)
    under_sqrt = dx**2 - 4 * k * (k + dy)

    if under_sqrt < 0:
        return None # Because we cannot square root negative numbers
    
    t1 = (dx + np.sqrt(under_sqrt)) / (2 * k) # tan(θ) high arc
    t2 = (dx - np.sqrt(under_sqrt)) / (2 * k) # tan(θ) low arc

    angle_1 = np.degrees(np.arctan(t1))
    angle_2 = np.degrees(np.arctan(t2))

    return tuple(sorted(angle_1, angle_2))

def pirate_shooting_arc(start, angle_deg, v, dt=0.17, g=9.81, max_steps=400):
    x, y = start
    theta = np.radians(angle_deg)

    vx = v * np.cos(theta)
    vy = -v * np.sin(theta) # "up" is negative vy

    pts = []

    for i in range(max_steps):
        pts.append((x, y))

        x += vx * dt # dt is the time step
        vy += g * dt
        y += vy * dt

        if y > HEIGHT or x <= 0 or x >= WIDTH:
            pts.append((x, min(y, HEIGHT)))
            break
    
    return pts

def simulate_pirate_shot():
    global selected_weapon, pirate_r_guess

     # Choose pirate bullet image based on player weapon
    if selected_weapon == "parrot":
        bullet_image = "weapon_parrot_bullet"
    elif selected_weapon == "cannon":
        bullet_image = "weapon_cannon_bullet"
    elif selected_weapon == "blunderbuss":
        bullet_image = "weapon_blunderbuss_bullet"
    else:
        bullet_image = "weapon_cannon_bullet"  # Fallback

    start = (pirate_cannon_x, pirate_cannon_y)

    target_x = pirate_r_guess * WIDTH
    target = (target_x, HEIGHT)

    pirate_speed = 65 # Fixed velocity or power for the pirate

    angles = pirate_shooting_angle(start, target, v=pirate_speed, g=9.81)

    if angles is None:
        angle_to_use = 45
    else:
        angle_to_use = angles[0]

    points = pirate_shooting_arc(start, angle_to_use, v=pirate_speed, dt=0.17, g=9.81)

    pirate_cannon_bullets.append({
        "Coordinates": points, # Coordinates
        "Position": 0, # Current position
        "Actor": ActorLike(bullet_image)
    })


# Cannon Game
def cannon_game():
    weapon_cannon.draw(screen)

    if cannon_shoot and not cannon_bullets:
        draw_cannon_arc()

    weapon_cannon.angle = -angle_deg

    # Displaying the angle and power variables to the player
    draw_multiline_text(
        screen,
        f"{deg_to_show}°\nPower: {power}",
        font_18,
        (255, 255, 255),
        (weapon_cannon.x - 120, weapon_cannon.y + 10)
    )

    # Draw cannon bullets
    for i in cannon_bullets:
        i["Actor"].draw(screen)

    # Telling the player how to shoot
    draw_multiline_text(
        screen,
        "Press SPACE to shoot",
        font_25,
        (255, 255, 255),
        (WIDTH // 2 - 100, HEIGHT -40)
    )

    if shot_message:
        draw_multiline_text(
            screen,
            shot_message,
            font_22,
            (255, 255, 255),
            (30, 450)
        )

    draw_multiline_text(
        screen,
        f"Rounds left: {rounds_left}",
        font_26,
        (255, 255, 255),
        (60, 20)
    )


def blunderbuss_game():
    global blunderbuss_aim_angle_deg

    shake_angle_deg = 0

    if blunderbuss_shoot and not blunderbuss_bullets:
        shake_angle_deg = random.uniform(-2, 2) # 2° wiggle
    
    # This is the angle we show
    blunderbuss_aim_angle_deg = blunderbuss_angle_deg + shake_angle_deg

    # Rotate the bluderbuss with the shaken angle
    weapon_blunderbuss.angle = -blunderbuss_aim_angle_deg

    # Or we can avoid having the Blunderbuss itself being shakne  and just have the aim shaken by:
    #weapon_blunderbuss.angle = -blunderbuss_angle_deg

    weapon_blunderbuss.draw(screen)

    if blunderbuss_shoot and not blunderbuss_bullets:
        draw_blunderbuss_arc(shake_angle_deg)

    # Displaying the angle and power variables to the player
    draw_multiline_text(
        screen,
        f"{blunderbuss_angle_deg}°\nPower: {power}",
        font_18,
        (255, 255, 255),
        (weapon_blunderbuss.x - 120, weapon_blunderbuss.y + 10)
    )

    # Draw Blunderbuss bullets
    for i in blunderbuss_bullets:
        i["Actor"].draw(screen)

    # Telling the player how to shoot
    draw_multiline_text(
        screen,
        "Press SPACE to shoot",
        font_25,
        (255, 255, 255),
        (WIDTH // 2 - 100, HEIGHT -40)
    )

    if shot_message:
        draw_multiline_text(
            screen,
            shot_message,
            font_22,
            (255, 255, 255),
            (30, 450)
        )

    draw_multiline_text(
        screen,
        f"Rounds left: {rounds_left}",
        font_26,
        (255, 255, 255),
        (60, 20)
    )


def calibrate_blood():
    global blood
    
    blood = booty / pirate_current_r * (1 - (1 + pirate_current_r)**(-plunders)) # Here we're making sure that the Pirate's r location will make the NPV = 0 by fitting the blood to the right value

    return blood

#Reset game function
def reset_game():
    global game_over, current_turn, rounds_left, shot_message
    global parrot_bullets, cannon_bullets, blunderbuss_bullets
    global parrot_shoot, cannon_shoot, blunderbuss_shoot
    global pirate_has_acted, pending_pirate_turn, pending_player_turn, midpoint
    global pirate_r_guess, player_turn_message_timer, pirate_message_timer
    global pirate_current_r, player_current_r, capital_g

    # Resetting the game state
    game_over = False
    current_turn = "Player"
    rounds_left = plunders
    shot_message = ""
    parrot_shoot = True
    cannon_shoot = True
    blunderbuss_shoot = True

    # Clearing active bullets
    parrot_bullets.clear()
    cannon_bullets.clear()
    blunderbuss_bullets.clear()
    pirate_cannon_bullets.clear()
    pirate_arc_points.clear()


    # Here we reset our opposing pirate logic
    pirate_has_acted = False
    pending_pirate_turn = False
    pending_player_turn = False
    pirate_r_guess = None
    midpoint[0] = 0
    midpoint[1] = 1.0
    capital_g = 0.0

    # Reseting timers
    pirate_turn_message_timer = 0.0
    pirate_message_timer = 0.0

    pirate_current_r = round(random.choice(pirate_r_list), 2)
    player_current_r = random.choice(pirate_r_list)

    calibrate_blood()

    pirate_r_guess = round(random.uniform(0.2, 0.7), 2)

# Layout buttons
def layout_menu():
    win_w, win_h = screen.get_size()
    btn_play.pos = (300, 500)
    btn_instructions.pos = (500, 500)
    btn_settings.pos = (win_w - 50, win_h - 55)

    btn_vol_up.pos = (win_w - 145, win_h - 215)
    btn_vol_down.pos = (btn_vol_up.x, btn_vol_up.y + 55)
    btn_vol_mute.pos = (btn_vol_up.x, btn_vol_down.y + 55)

# Selection Screen buttons
def selections_menu():
    btn_parrot.pos = (700, 175)
    btn_cannon.pos = (700, 325)
    btn_blunderbuss.pos = (700, 475)
    btn_easy.pos = (450, 175)
    btn_medium.pos = (450, 325)
    btn_hard.pos = (450, 475)
    btn_back.pos = (20, 20)
    plunders_box_img.pos = (200, 300)
    btn_start.pos = (150, 520) # selections screen

# In Game Buttons and Actors
def game_menu():
    player_ship.pos = (700, 400)
    btn_back.pos = (20, 20)

def win_menu():
    btn_back.pos = (20, 20)

def loss_menu():
    btn_back.pos = (20, 20)

def tie_menu():
    btn_back.pos = (20, 20)

def instructions_menu():
    btn_back.pos = (15, 20)


# Draw background image to cover screen
def draw_background_cover(image):
    win_w, win_h = screen.get_size()
    img_w, img_h = image.get_size()

    scale = max(win_w / img_w, win_h / img_h)

    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    scaled = pygame.transform.smoothscale(image, (new_w, new_h))
    x = (win_w - new_w) // 2
    y = (win_h - new_h) // 2

    screen.blit(scaled, (x, y))


# Main Menu draw
def draw_menu():
    draw_background_cover(pygame.image.load("images/welcome_background.jpg").convert())
    layout_menu()
    btn_play.draw(screen)
    btn_instructions.draw(screen)
    btn_settings.draw(screen)

    if settings_open:
        win_w, win_h = screen.get_size()
        draw_settings_panel(win_w, win_h)

# Settings panel
def draw_settings_panel(win_w, win_h):
    panel_rect = pygame.Rect(
        int(win_w - 220), # x location
        int(win_h - 290), # y location
        int(150), # Width
        int(220), # Height
    )

    pygame.draw.rect(screen, (255,255,255), panel_rect, 0) # (255, 255, 255) is for the white colour
    pygame.draw.rect(screen, (0, 0, 0), panel_rect, width=1)

    draw_multiline_text(
        screen,
        "Settings",
        font_30,
        (0, 0, 0),
        (panel_rect.centerx - 45, panel_rect.top + 10)
    )

    btn_vol_up.draw(screen)
    btn_vol_down.draw(screen)
    btn_vol_mute.draw(screen)

# Weapon selection and difficulty selection screen
def draw_selections():
    selections_menu()
    draw_background_cover(pygame.image.load("images/selections_background.jpg").convert())
    pirate_rect = pygame.Rect(
        plunders_box_img.left + 150,
        plunders_box_img.top + 230,
        plunders_box_img.width // 15,
        plunders_box_img.height // 5
    )
    pygame.draw.rect(screen, (255,255,255), pirate_rect, 0)

    # Circle draws behind weapon under here
    pygame.draw.circle(screen, (255, 255, 255), btn_parrot.pos, 60, width=0)
    pygame.draw.circle(screen, (255, 255, 255), btn_cannon.pos, 60, width=0)
    pygame.draw.circle(screen, (255, 255, 255), btn_blunderbuss.pos, 60, width=0)
    if selected_weapon == "parrot":
        pygame.draw.circle(screen, (255, 165, 0), btn_parrot.pos, 60, width=0)
    elif selected_weapon == "cannon":
        pygame.draw.circle(screen, (255, 165, 0), btn_cannon.pos, 60, width=0)
    elif selected_weapon == "blunderbuss":
        pygame.draw.circle(screen, (255, 165, 0), btn_blunderbuss.pos, 60, width=0)


    btn_parrot.draw(screen)
    btn_cannon.draw(screen)
    btn_blunderbuss.draw(screen)
    btn_easy.draw(screen)
    btn_medium.draw(screen)
    btn_hard.draw(screen)
    btn_back.draw(screen)
    plunders_box_img.draw(screen)

    border_color = (255, 165, 0) if plunder_box_active else (0, 0, 0) # (255, 165, 0) for orange; (0, 0, 0) for black

    pygame.draw.rect(screen, (255,255,255), plunders_box, 0) # Drawing the Plunders text box
    pygame.draw.rect(screen, border_color, plunders_box, width=1) # Drawing the border of the Plunders text box

    if easy_box_active:
        pygame.draw.rect(screen, (0, 0, 0), easy_box, width=5)
    if medium_box_active:
        pygame.draw.rect(screen, (0, 0, 0), medium_box, width=5)
    if hard_box_active:
        pygame.draw.rect(screen, (0, 0, 0), hard_box, width=5)

    draw_multiline_text(
        screen,
        plunder_text,
        font_32,
        (0, 0, 0),
        (plunders_box.x + 5, plunders_box.y + 5)
    )

    draw_multiline_text(
        screen,
        "Press ENTER",
        font_25,
        (0, 0, 0),
        (197, 320)
    )

    if selected_weapon != None and selected_level != None: # Button pops up when weapon is picked
        btn_start.draw(screen)

# Game screen
def draw_game():
    draw_background_cover(pygame.image.load("images/game_background.jpg").convert())
    game_menu()
    player_ship.draw(screen)
    btn_back.draw(screen)

    # DRAW PIRATE CANNON SHOTS
    for bullet in pirate_cannon_bullets:
        if bullet["Position"] < len(bullet["Coordinates"]):
            bullet["Actor"].draw(screen)

    if selected_weapon == "parrot":
        weapon_parrot.draw(screen)

        # Telling the player how to shoot
        draw_multiline_text(
            screen,
            "Press SPACE to shoot",
            font_25,
            (255, 255, 255),
            (WIDTH // 2 - 100, HEIGHT - 40)
        )

        draw_multiline_text(
            screen,
            f"Parrot's Coordinates: {weapon_parrot.x}",
            font_18,
            (255, 255, 255),
            (WIDTH - 225, HEIGHT - 35)
        )

        # Game info to display
        draw_multiline_text(
            screen,
            f"Rounds left: {rounds_left}",
            font_26,
            (255, 255, 255),
            (60, 20)
        )

        # Display results of shot for parrot
        if shot_message:
            draw_multiline_text(
                screen,
                shot_message,
                font_22,
                (255, 255, 255),
                (30, 450)
            )

        for i in parrot_bullets:
            i["Actor"].draw(screen)
            
    if selected_weapon == "cannon":
        cannon_game()        
    if selected_weapon == "blunderbuss":
        blunderbuss_game()

# Win screen
def draw_win():
    draw_background_cover(pygame.image.load("images/win_background.jpg").convert())
    win_menu()
    btn_back.draw(screen)

# Win screen
def draw_lose():
    draw_background_cover(pygame.image.load("images/loss_background.jpg").convert())
    loss_menu()
    btn_back.draw(screen)

# Draw screen (as in tie)
def draw_tie():
    draw_background_cover(pygame.image.load("images/draw_background.jpg").convert())
    tie_menu()
    btn_back.draw(screen)

# Instructions screen
def draw_instructions():
    draw_background_cover(pygame.image.load("images/instructions_screen.jpg").convert())
    instructions_menu()
    btn_back.draw(screen)

# Main draw
def draw():
    if current_screen == "Menu":
        draw_menu()
    if current_screen == "Selections":
        draw_selections()
    if current_screen == "Win":
        draw_win()
    if current_screen == "Lose":
        draw_lose()
    elif current_screen == "Game":
        draw_game()
    if current_screen == "Instructions":
        draw_instructions()
    if current_screen == "Draw":
        draw_tie()

# Click handling
def on_mouse_down(pos):
    global current_screen, settings_open, volume, muted, selected_weapon, selected_level
    global plunder_box_active, easy_box_active, medium_box_active, hard_box_active
    global music_started

    if not music_started:
        pygame.mixer.music.load("music/sea_shanty_2.mp3")
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play(-1) # 0: Play once; 1: Loop 1 extra time (total plays: 2); -1: Loop indefinitely until you stop it
        
        music_started = True

    if current_screen == "Menu":

        if btn_play.collidepoint(pos):
            current_screen = "Selections"
        # Optional: Uncomment line below if you want music to stop when leaving menu
        # music.stop()

        elif btn_instructions.collidepoint(pos):
            current_screen = 'Instructions'

        elif btn_settings.collidepoint(pos):
            settings_open = not settings_open

        if settings_open:

            if btn_vol_up.collidepoint(pos):
                volume = min(1.0, volume + 0.1)
                if not muted: # Only updates when not muted
                    pygame.mixer.music.set_volume(volume)
                print("Volume up:", round(volume, 2))

            elif btn_vol_down.collidepoint(pos):
                volume = max(0.0, volume - 0.1)
                if not muted:
                    pygame.mixer.music.set_volume(volume)
                print("Volume down:", round(volume, 2))

            elif btn_vol_mute.collidepoint(pos):
                muted = not muted
                if muted:
                    pygame.mixer.music.set_volume(0)
                else:
                    pygame.mixer.music.set_volume(volume)
                print("Muted:", muted)


# Selections click handling
    elif current_screen == "Selections":
    
        # This is for the Plunders text box
        if plunders_box.collidepoint(pos):
            plunder_box_active = True
        else:
            plunder_box_active = False

        # Highlighting the level buttons
        if easy_box.collidepoint(pos):
            easy_box_active = True
            medium_box_active = False
            hard_box_active = False

        if medium_box.collidepoint(pos):
            medium_box_active = True
            easy_box_active = False
            hard_box_active = False

        if hard_box.collidepoint(pos):
            hard_box_active = True
            medium_box_active = False
            easy_box_active = False
        
        if (btn_back.collidepoint(pos)):
            current_screen = "Menu"
        
        if btn_parrot.collidepoint(pos):
            selected_weapon = "parrot"
            #current_screen = "Game"
            weapon_parrot.pos = (450, 100)

        elif btn_cannon.collidepoint(pos):
            selected_weapon = "cannon"
            #current_screen = "Game"
            weapon_cannon.pos = (650, 540)

        elif btn_blunderbuss.collidepoint(pos):
            selected_weapon = "blunderbuss"
            #current_screen = "Game"
            weapon_blunderbuss.pos = (650, 520)

        if btn_start.collidepoint(pos):
            if selected_weapon != None and selected_level != None:
                current_screen = "Game"

        if btn_easy.collidepoint(pos):
            selected_level = "Easy"
        elif btn_medium.collidepoint(pos):
            selected_level = "Medium"
        elif btn_hard.collidepoint(pos):
            selected_level = "Hard"

    if current_screen == "Game" and btn_back.collidepoint(pos):
        reset_game() # Calls upon function and resets variables
        current_screen = "Selections"

    if current_screen == "Win" and btn_back.collidepoint(pos):
        reset_game() # Calls upon function and resets variables
        current_screen = "Selections"

    if current_screen == "Lose" and btn_back.collidepoint(pos):
        reset_game() # Calls upon function and resets variables
        current_screen = "Selections"
    if current_screen == "Instructions" and btn_back.collidepoint(pos):
        reset_game() # Calls upon function and resets variables
        current_screen = "Menu"

    if current_screen == "Draw" and btn_back.collidepoint(pos):
        reset_game() # Calls upon function and resets variables
        current_screen = "Selections"
    

def update():
    global rounds_left, current_turn, game_over, pirate_r_guess, current_screen, difficulty_sd
    global angle_deg, deg_to_show, power, parrot_shoot, cannon_shoot, blunderbuss_shoot, blunderbuss_angle_deg
    global landing_x, player_r_guess, shot_message, last_player_message
    global pirate_has_acted, pirate_message_timer
    global player_message_timer, pending_pirate_turn, pending_player_turn, player_turn_message_timer

    g = 9.81
    t = 0.17 # Time per frames in seconds

    # Updating the pirate's cannon animation
    for bullet in pirate_cannon_bullets:
        bullet["Position"] += 1

        if bullet["Position"] < len(bullet["Coordinates"]):
            x, y = bullet["Coordinates"][bullet["Position"]]
            bullet["Actor"].pos = (x, y)
        else:
            pirate_cannon_bullets.remove(bullet)


    if pending_pirate_turn and not game_over:
        player_message_timer -= t

        if player_message_timer <= 0:
            pending_pirate_turn = False
            current_turn = "Pirate"
            pirate_has_acted = False
            pirate_message_timer = 0.0

    if pending_player_turn and not game_over:
        player_turn_message_timer -= t

        if player_turn_message_timer <= 0:
            pending_player_turn = False
            current_turn = "Player"
            shot_message = "ARR, It's Your Turn Captain!" + last_player_message[24:]

    # Pirate game loop using logic from cannon
    if current_screen == 'Game' and selected_weapon == 'parrot':

        if not game_over and current_turn == "Pirate":
            # If the Pirate did not play yet
            if not pirate_has_acted:

                if pirate_r_guess is None:
                    pirate_r_guess = round(random.uniform(0.2, 0.7), 2)


                # Took bisection logic from cannon
                if selected_level == "Easy":
                    difficulty_sd = 0.05
                    pirate_r_guess = (midpoint[0] + midpoint[1]) / 2
                    if npv_zero(pirate_r_guess, player_current_r, plunders, booty)[2] > 0:
                        midpoint[0] = pirate_r_guess
                    else:
                        midpoint[1] = pirate_r_guess

                elif selected_level == "Medium":
                    difficulty_sd = 0.03
                    x = npv_zero(pirate_r_guess, player_current_r, plunders, booty)[1]
                    pirate_r_guess = max(0.01, min(0.99, x)) # We're doing this to make sure that Newton–Raphson Method doesn't produce value outside of our range

                elif selected_level == "Hard":
                    difficulty_sd = 0.01

                    pirate_r_guess, blood_local, npv_val, npv_der = pirate_hard_mode(
                        pirate_r_guess, player_current_r, plunders, booty, method="adam" # Or adagrad if we want to try it
                    )

                # PIRATE CANNON ANIMATION
                # Convert pirate r-guess to a landing X coordinate
                simulate_pirate_shot()

                # Check if Pirate Won
                if abs(round(pirate_r_guess, 2) - player_current_r) < 0.01:
                    shot_message = (f"The Pirate Hit You! You Lost!\nPirate guessed r = {pirate_r_guess:.2f}")
                    game_over = True
                    current_screen = 'Lose'
                    
                    """
                    # Updating the memory CSV file
                    blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                    calc_error = pirate_r_guess - player_current_r
                    calc_loss = 0.5 * npv_val**2
                    calc_gradient = npv_val * npv_der

                    across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                    """
                else:
                    rounds_left -= 1
                    if rounds_left <= 0:
                        shot_message = (f"Pirate guessed r = {pirate_r_guess:.2f}\nOut of Plunders! It's a Draw! You Survived!!")
                        game_over = True
                        current_screen = 'Draw'

                        """
                        # Updating the memory CSV file
                        blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                        calc_error = pirate_r_guess - player_current_r
                        calc_loss = 0.5 * npv_val**2
                        calc_gradient = npv_val * npv_der

                        across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                        """
                    else:
                        shot_message = (
                            f"Phew... The Pirate Missed Us, Captain!\n"
                            f"\nPirate's r value's shot: {pirate_r_guess:.6f}"
                            f"\nOur r value: {player_current_r}"
                        )

                pirate_message_timer = 15.0
                pirate_has_acted = True

            else:
                pirate_message_timer -= t
                if pirate_message_timer <= 0 and not game_over:
                    pirate_has_acted = False
                    pending_player_turn = True
                    player_turn_message_timer = 10.0
                    current_turn = "Between"

       # How the parrote moves
        if current_turn == "Player" and game_over == False:  # Only move if it's player turn
            if keys[pygame.K_LEFT]:
                weapon_parrot.x -= 7
            if keys[pygame.K_RIGHT]:
                weapon_parrot.x += 7

            # Keep the parrot on screen
            if weapon_parrot.left < 0:
                weapon_parrot.left = 0
            if weapon_parrot.right > WIDTH:
                weapon_parrot.right = WIDTH

            if keys[pygame.K_SPACE] and parrot_shoot and len(parrot_bullets) == 0:
                start_x = weapon_parrot.x
                start_y = weapon_parrot.y + 50 # The 50 will start the drop below the parrot
                parrot_dic = {
                    "Actor": ActorLike("weapon_parrot_bullet"),
                    "x": start_x,
                    "y": start_y,
                    "vy": 0 # We start with a velocity of 0
                }
                parrot_bullets.append(parrot_dic)
                parrot_shoot = False # Stops parrot from multiple shots

        # Resetting parrot_shoot
        if not keys[pygame.K_SPACE]:
            parrot_shoot = True

        for i in parrot_bullets:
            i["vy"] += g * t # gravity is added to the vertical speed, increases with time
            i["y"] += i["vy"] * t # Updates the vertical position to move with time
            i["Actor"].pos = (i["x"], i["y"]) # Draws our bomb at the new cooridinates

            # Hit marker check on ground
            if i["y"] >= HEIGHT:
                landing_x = i["x"]

                # Took this math from cannon
                player_r_guess = max(0, min(1, landing_x / WIDTH))
                player_r_guess = round(player_r_guess, 2)

                if abs(player_r_guess - pirate_current_r) < difficulty_sd:
                    shot_message = "ARR You Won!"
                    game_over = True
                    current_screen = 'Win'

                    """
                    # Updating the memory CSV file
                    blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                    calc_error = pirate_r_guess - player_current_r
                    calc_loss = 0.5 * npv_val**2
                    calc_gradient = npv_val * npv_der

                    across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                    """
                else:
                    last_player_message = (
                        "You Missed Him, Captain!\n"
                        f"\nBlood (Initial Cost): {blood:.2f}"
                        f"\nBooty (CFs): {booty}"
                        f"\nYour r value's shot: {player_r_guess}"
                        f"\nYour NPV: {npv_zero(player_r_guess, pirate_current_r, plunders, booty)[2]:.2f}"
                    )
                    shot_message = last_player_message
                    current_turn = "Resolving"
                    player_message_timer = 30.0
                    pending_pirate_turn = True

                parrot_bullets.remove(i)  # Remove bullet after it hits ground


    # The Cannon Game
    if current_screen == "Game" and selected_weapon == "cannon":
        if not game_over and current_turn == "Pirate":

            # If the Pirate did not play yet    
            if not pirate_has_acted:

                if pirate_r_guess is None:
                    pirate_r_guess = round(random.uniform(0.2, 0.7), 2)
                
                # Setting the Easy Level Game with the Bisection Method
                if selected_level == "Easy":
                    difficulty_sd = 0.05
                    pirate_r_guess = (midpoint[0] + midpoint[1]) / 2
                    if npv_zero(pirate_r_guess, player_current_r, plunders, booty)[2] > 0:
                        midpoint[0] = pirate_r_guess
                    else:
                        midpoint[1] = pirate_r_guess

                elif selected_level == "Medium":
                    difficulty_sd = 0.03
                    x = npv_zero(pirate_r_guess, player_current_r, plunders, booty)[1]
                    pirate_r_guess = max(0.01, min(0.99, x)) # We're doing this to make sure that Newton–Raphson Method doesn't produce value outside of our range

                elif selected_level == "Hard":
                    difficulty_sd = 0.01

                    pirate_r_guess, blood_local, npv_val, npv_der = pirate_hard_mode(
                        pirate_r_guess, player_current_r, plunders, booty, method="adam" # Or adagrad if we want to try it
                    )

                simulate_pirate_shot()


                if abs(round(pirate_r_guess, 2) - player_current_r) < 0.01:
                    shot_message = (
                        "The Pirate Hit You! You Lost!"
                        f"\nPirate guessed r = {pirate_r_guess:.2f}"
                    )
                    game_over = True
                    current_screen = "Lose"

                    """
                    # Updating the memory CSV file
                    blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                    calc_error = pirate_r_guess - player_current_r
                    calc_loss = 0.5 * npv_val**2
                    calc_gradient = npv_val * npv_der

                    across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                    """
                else:
                    rounds_left -= 1

                    if rounds_left <= 0:
                        shot_message = (
                            f"Pirate guessed r = {pirate_r_guess:.2f}"
                            "\nOut of Plunders! It's a Draw! You Survived!!"
                        )
                        game_over = True
                        current_screen = "Draw"

                        """
                        # Updating the memory CSV file
                        blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                        calc_error = pirate_r_guess - player_current_r
                        calc_loss = 0.5 * npv_val**2
                        calc_gradient = npv_val * npv_der

                        across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                        """
                    else:
                        shot_message = (
                            f"Phew... The Pirate Missed Us, Captain!\n" 
                            f"\nPirate's r value's shot: {pirate_r_guess:.6f}"
                            f"\nOur r value: {player_current_r}"
                        )

                pirate_message_timer = 15.0
                pirate_has_acted = True

            else:
                pirate_message_timer -= t

                if pirate_message_timer <= 0 and not game_over:
                    pirate_has_acted = False
                    pending_player_turn = True
                    player_turn_message_timer = 10.0
                    current_turn = "Between" # Freezing shooting


        if not game_over and current_turn == "Player":

            # Angle Controls
            if keys[pygame.K_LEFT]:
                angle_deg -= 1
                deg_to_show -= 1
        
            if keys[pygame.K_RIGHT]:
                angle_deg += 1
                deg_to_show += 1
        
            angle_deg = max(-25, min(65, angle_deg)) # Making sure that -25 <= angle actual degree <= 65
            deg_to_show = max(0, min(90, deg_to_show)) # Making sure that 0 <= base angle degree <= 90

            # Power Controls
            if keys[pygame.K_UP]:
                power += 1

            if keys[pygame.K_DOWN]:
                power -= 1

            power = max(0, min(100, power)) # Limiting power 0 <= Power <= 100

            
            v = power
            deg_to_show_rad = np.deg2rad(deg_to_show)

            if keys[pygame.K_SPACE] and cannon_shoot and len(cannon_bullets) == 0 and not game_over and current_turn == "Player":
            
                start_x, start_y = barrel_tip(cannon_offset_x, cannon_offset_y, weapon_cannon, deg_to_show, angle_deg)

                end_x = -v * np.cos(deg_to_show_rad)
                end_y = -v * np.sin(deg_to_show_rad)

                bullet_dic_cannon = {
                    "Actor": cannon_bullet,
                    "x": start_x,
                    "y": start_y,
                    "Ending x": end_x,
                    "Ending y": end_y
                }
                bullet_dic_cannon["Actor"].pos = (start_x, start_y)
                cannon_bullets.append(bullet_dic_cannon)

                cannon_shoot = False # Waiting for the Spacebar to be released before making another shot, otherwise it will be like a machine gun.

            # Reseting cannon_shoot after Spacebar is released.
            if not keys[pygame.K_SPACE]:
                cannon_shoot = True

            for i in cannon_bullets:
                i["x"] += i["Ending x"] * t
                i["Ending y"] += g * t # Gravity
                i["y"] += i["Ending y"] * t
                i["Actor"].pos = (i["x"], i["y"])

                if i["y"] >= HEIGHT or i["x"] <= 0:
                    landing_x = i["x"] # We're saving the x value when x lands

                    player_r_guess = max(0, min(1, landing_x / WIDTH)) # The r guess the player made can only be between 0 and 1 and is calculated by getting its proportion to the entire screen.
                    player_r_guess = round(player_r_guess, 2)

                    if abs(player_r_guess - pirate_current_r) < difficulty_sd:
                        shot_message = "ARR You Won!"
                        game_over = True
                        current_screen = 'Win'
                    else:
                        last_player_message = (
                            "You Missed Him, Captain!\n"
                            f"\nBlood (Initial Cost): {blood:.2f}"
                            f"\nBooty (CFs): {booty}"
                            f"\nYour r value's shot: {player_r_guess}"
                            f"\nYour NPV: {npv_zero(player_r_guess, pirate_current_r, plunders, booty)[2]:.2f}"
                        )
                        shot_message = last_player_message
                        current_turn = "Resolving" # We're doing this so that the player wouldn't be able to shoot after he made a shot
                        player_message_timer = 30.0
                        pending_pirate_turn = True

                    cannon_bullets.remove(i)

    # The Blunderbuss Game
    if current_screen == "Game" and selected_weapon == "blunderbuss":
        if not game_over and current_turn == "Pirate":

            # If the Pirate did not play yet    
            if not pirate_has_acted:

                if pirate_r_guess is None:
                    pirate_r_guess = round(random.uniform(0.2, 0.7), 2)

                # Setting the Easy Level Game with the Bisection Method
                if selected_level == "Easy":
                    difficulty_sd = 0.05
                    pirate_r_guess = (midpoint[0] + midpoint[1]) / 2
                    if npv_zero(pirate_r_guess, player_current_r, plunders, booty)[2] > 0:
                        midpoint[0] = pirate_r_guess
                    else:
                        midpoint[1] = pirate_r_guess

                elif selected_level == "Medium":
                    difficulty_sd = 0.03
                    x = npv_zero(pirate_r_guess, player_current_r, plunders, booty)[1]
                    pirate_r_guess = max(0.01, min(0.99, x)) # We're doing this to make sure that Newton–Raphson Method doesn't produce value outside of our range

                elif selected_level == "Hard":
                    difficulty_sd = 0.01

                    pirate_r_guess, blood_local, npv_val, npv_der = pirate_hard_mode(
                        pirate_r_guess, player_current_r, plunders, booty, method="adam" # Or adagrad if we want to try it
                    )

                simulate_pirate_shot()


                if abs(round(pirate_r_guess, 2) - player_current_r) < 0.01:
                    shot_message = (
                        "The Pirate Hit You! You Lost!"
                    )
                    game_over = True
                    current_screen = 'Lose'

                    """
                    # Updating the memory CSV file
                    blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                    calc_error = pirate_r_guess - player_current_r
                    calc_loss = 0.5 * npv_val**2
                    calc_gradient = npv_val * npv_der

                    across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                    """
                else:
                    rounds_left -= 1

                    if rounds_left <= 0:
                        shot_message = (
                            f"Pirate guessed r = {pirate_r_guess:.2f}"
                            "\nOut of Plunders! It's a Draw! You Survived!!"
                        )
                        game_over = True
                        current_screen = 'Draw'

                        """
                        # Updating the memory CSV file
                        blood_local, x, npv_val, npv_der, newton_step = npv_zero(pirate_r_guess, player_current_r, plunders, booty)
                        calc_error = pirate_r_guess - player_current_r
                        calc_loss = 0.5 * npv_val**2
                        calc_gradient = npv_val * npv_der

                        across_game_optimization(booty, blood_local, plunders, pirate_r_guess, player_current_r, npv_val, calc_error, calc_loss, calc_gradient)
                        """
                    else:
                        shot_message = (
                            f"Phew... The Pirate Missed Us, Captain!\n"
                            f"\nPirate's r value's shot: {pirate_r_guess:.6f}"
                            f"\nOur r value: {player_current_r}"
                        )

                pirate_message_timer = 15.0
                pirate_has_acted = True

            else:
                pirate_message_timer -= t

                if pirate_message_timer <= 0 and not game_over:
                    pirate_has_acted = False
                    pending_player_turn = True
                    player_turn_message_timer = 10.0
                    current_turn = "Between" # Freezing shooting


        if not game_over and current_turn == "Player":

            # Angle Controls
            if keys[pygame.K_LEFT]:
                blunderbuss_angle_deg -= 1
        
            if keys[pygame.K_RIGHT]:
                blunderbuss_angle_deg += 1
        
            blunderbuss_angle_deg = max(0, min(90, blunderbuss_angle_deg)) # Making sure that 0 <= base angle degree <= 90

            # Power Controls
            if keys[pygame.K_UP]:
                power += 1

            if keys[pygame.K_DOWN]:
                power -= 1

            power = max(0, min(100, power)) # Limiting power 0 <= Power <= 100

            
            v = power
            angle_rad = np.deg2rad(blunderbuss_aim_angle_deg)

            if keys[pygame.K_SPACE] and blunderbuss_shoot and len(blunderbuss_bullets) == 0 and not game_over and current_turn == "Player":

                start_x, start_y = barrel_tip(blunderbuss_offset_x, blunderbuss_offset_y, weapon_blunderbuss, blunderbuss_aim_angle_deg)

                end_x = -v * np.cos(angle_rad)
                end_y = -v * np.sin(angle_rad)

                bullet_dic_blunderbuss = {
                    "Actor": blunderbuss_bullet,
                    "x": start_x,
                    "y": start_y,
                    "Ending x": end_x,
                    "Ending y": end_y
                }
                bullet_dic_blunderbuss["Actor"].pos = (start_x, start_y)
                blunderbuss_bullets.append(bullet_dic_blunderbuss)

                blunderbuss_shoot = False # Waiting for the Spacebar to be released before making another shot, otherwise it will be like a machine gun.

            # Reseting blunderbuss_shoot after Spacebar is released.
            if not keys[pygame.K_SPACE]:
                blunderbuss_shoot = True

            for i in blunderbuss_bullets:
                i["x"] += i["Ending x"] * t
                i["Ending y"] += g * t # Gravity
                i["y"] += i["Ending y"] * t
                i["Actor"].pos = (i["x"], i["y"])

                if i["y"] >= HEIGHT or i["x"] <= 0:
                    landing_x = i["x"] # We're saving the x value when x lands

                    player_r_guess = max(0, min(1, landing_x / WIDTH)) # The r guess the player made can only be between 0 and 1 and is calculated by getting its proportion to the entire screen.
                    player_r_guess = round(player_r_guess, 2)

                    blunderbuss_sd = round(random.uniform(player_r_guess - 0.02, player_r_guess + 0.02), 4)

                    if abs(blunderbuss_sd - pirate_current_r) < difficulty_sd:
                        shot_message = "ARR You Won!"
                        game_over = True
                        current_screen = 'Win'
                    else:
                        last_player_message = (
                            "You Missed Him, Captain!\n"
                            f"\nBlood (Initial Cost): {blood:.2f}"
                            f"\nBooty (CFs): {booty}"
                            f"\nYour r value's shot: {player_r_guess}"
                            f"\nYour NPV: {npv_zero(player_r_guess, pirate_current_r, plunders, booty)[2]:.2f}"
                        )
                        shot_message = last_player_message
                        current_turn = "Resolving" # We're doing this so that the player wouldn't be able to shoot after he made a shot
                        player_message_timer = 30.0
                        pending_pirate_turn = True

                    blunderbuss_bullets.remove(i)  


# This function is for handling the Plunders text box
def on_key_down(key):
    global plunder_text, plunder_box_active, plunders
    global rounds_left, current_turn, game_over, pirate_r_guess, pirate_current_r, player_current_r

    if not plunder_box_active:
        return
    
    # If the user wants to erased the values he typed
    if key == pygame.K_BACKSPACE:
        plunder_text = plunder_text[:-1]
        return

    # Handling the plunder's text box
    if key == pygame.K_RETURN: # If the player presses ENTER

        if plunder_text.isdigit() and int(plunder_text) > 0:
            plunders = int(plunder_text)
        else:
            plunders = 1 # Defaulting to pluders of 1 if an empty box was entered
        
        #plunder_text = "" # Clear box after enter
        plunder_box_active = False
        midpoint[0], midpoint[1] = 0, 1.0
        pirate_current_r = round(random.choice(pirate_r_list), 2)
        player_current_r = random.choice(pirate_r_list)

        calibrate_blood()

        # Reset game state that depends on plunders
        current_turn = "Player"
        game_over = False
        pirate_r_guess = None
        rounds_left = plunders

        return
    
    # Defining the digits in a dictionary
    digit_map = {
        pygame.K_0: "0", pygame.K_1: "1", pygame.K_2: "2", pygame.K_3: "3", pygame.K_4: "4",
        pygame.K_5: "5", pygame.K_6: "6", pygame.K_7: "7", pygame.K_8: "8", pygame.K_9: "9",
        pygame.K_KP0: "0", pygame.K_KP1: "1", pygame.K_KP2: "2", pygame.K_KP3: "3", pygame.K_KP4: "4",
        pygame.K_KP5: "5", pygame.K_KP6: "6", pygame.K_KP7: "7", pygame.K_KP8: "8", pygame.K_KP9: "9",
    }

    # If the keyboard key that the user is pressing is in the dictionary then we add it to plunder_text
    if key in digit_map:
        plunder_text += digit_map[key]

        # Here we're limiting the figure number
        if len(plunder_text) > 3:
            plunder_text = plunder_text[:3]


if __name__ == "__main__":
    main()