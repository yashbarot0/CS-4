CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -lm
TARGET = multigrid

all: $(TARGET)

$(TARGET): multigrid.c
	$(CC) -o $(TARGET) multigrid.c $(CFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean