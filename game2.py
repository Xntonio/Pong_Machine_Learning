import pygame
import sys

# Inicializar pygame
pygame.init()

# Tamaño de la ventana
HEIGHT = 420
WIDTH = 800

# Dimensiones de la raqueta
PAD_WIDTH = 20
PAD_HEIGHT = 200

# Velocidad de la raqueta
PAD_VEL = 20

# Coordenadas de la raqueta
pad_x = WIDTH - PAD_WIDTH
pad_y = HEIGHT - PAD_HEIGHT
 
# Coordenadas de la pelota
ball_x = 0
ball_y = HEIGHT / 2

# Dirección de la pelota
ball_dir_x = 1
ball_dir_y = 1
BALL_VEL=15

# Tamaño de la pelota
BALL_SIZE = 10

# Puntuación
score = 0

# Game over
game_over = 0

# Crear la ventana
window = pygame.display.set_mode((WIDTH, HEIGHT))

# Bucle principal del juego
while True:
    # Manejar eventos
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    # Mover la raqueta con el teclado
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        pad_y -= PAD_VEL
    if keys[pygame.K_DOWN]:
        pad_y += PAD_VEL

    # Asegurarse de que la raqueta esté dentro de la ventana
    if pad_y <= 0:
        pad_y = 0
    elif pad_y >= HEIGHT - PAD_HEIGHT:
        pad_y = HEIGHT - PAD_HEIGHT

    # Mover la pelota
    ball_x += ball_dir_x*BALL_VEL
    ball_y += ball_dir_y*BALL_VEL

    # Verificar si la pelota ha tocado la raqueta
    if PAD_WIDTH+ball_x >= pad_x  and ball_y >= pad_y and ball_y <= pad_y + PAD_HEIGHT:
        score += 1
        ball_dir_x = -ball_dir_x

    # Verificar si la pelota ha tocado las paredes
    if ball_y <= 0 or ball_y >= HEIGHT-20:
        ball_dir_y = -ball_dir_y

    if ball_x <= 0:
        ball_dir_x=-ball_dir_x
    # Verificar si la pelota ha salido de la ventana
    if ball_x >= WIDTH:
        game_over = 1

    # Dibujar la raqueta y la pelota en la ventana
    window.fill((0, 0, 0))
    pygame.draw.rect(window, (255, 255, 255), (pad_x, pad_y, PAD_WIDTH, PAD_HEIGHT))
    pygame.draw.circle(window, (255, 255, 0), (int(ball_x + BALL_SIZE / 2), int(ball_y + BALL_SIZE)), radius=BALL_SIZE)

    # Actualizar la ventana
    pygame.display.update()

    # Mostrar la puntuación
    font = pygame.font.Font(None, 30)
    text = font.render("Score: " + str(score), 1, (255, 255, 255))
    window.blit(text, (0, 0))
    pygame.time.wait(50)

    # Verificar si el juego ha terminado
    if game_over:
        font = pygame.font.Font(None, 60)
        text = font.render("Game Over", 1, (255, 255, 255))
        window.blit(text, (WIDTH / 2 - 100, HEIGHT / 2))
        pygame.display.update()
        pygame.time.wait(3000)
        sys.exit()
