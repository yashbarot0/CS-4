CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -lm
TARGET = multigrid
SRC = multigrid.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) $(CFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all clean