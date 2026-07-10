# this assumes environment variable GAME is set to the game name, e.g. othello

.PHONY: check_directory

CFLAGS = -Wall -Werror -pedantic-errors -O3 -fPIC
#CFLAGS = -Wall -Werror -pedantic-errors -g -O0 -fPIC

include = -I./src/c -I./$(GAME)/src/c
main_builddir = ./build/
builddir = ./build/$(GAME)
	
all: aes_hash.o dict32.o game.o libgame.so libnetwork.so random.o MCTS_ucb.o libMCTS_ucb.so RMCTS.o libRMCTS.so | build
	cp ./src/python/*.py $(builddir) # generic python files
	cp ./$(GAME)/src/python/*.py $(builddir) # game-specific python files
	echo gamename=\'$(GAME)\' > $(builddir)/__init__.py
	touch $(builddir)/__init__.py

main_build:
	mkdir -p $(main_builddir)
	
build: main_build
	mkdir -p $(builddir)
	
aes_hash.o: ./src/c/aes_hash.c ./src/c/aes_hash.h | build
	gcc -o $(builddir)/aes_hash.o $(CFLAGS) $(include) -march=native -c ./src/c/aes_hash.c

dict32.o: ./src/c/dict32.c ./src/c/dict32.h ./src/c/aes_hash.h | build
	gcc -o $(builddir)/dict32.o $(CFLAGS) $(include) -c ./src/c/dict32.c

game.o: ./$(GAME)/src/c/game.c ./src/c/game.h | build
	gcc -o $(builddir)/game.o $(CFLAGS) $(include) -c ./$(GAME)/src/c/game.c

libgame.so: game.o | build
	gcc -o $(builddir)/libgame.so -shared $(builddir)/game.o

libnetwork.so: ./src/c/game_to_nnet.c game.o | build
	gcc -o $(builddir)/libnetwork.so -shared ./src/c/game_to_nnet.c $(builddir)/game.o

random.o: ./src/c/random.c ./src/c/game.h | build
	gcc -o $(builddir)/random.o $(CFLAGS) $(include) -c ./src/c/random.c

MCTS_ucb.o: ./src/c/MCTS_ucb.c ./src/c/MCTS_ucb.h ./src/c/dict32.h ./src/c/game.h | build
	gcc -o $(builddir)/MCTS_ucb.o $(CFLAGS) $(include) -c ./src/c/MCTS_ucb.c -pthread

libMCTS_ucb.so: MCTS_ucb.o game.o random.o dict32.o aes_hash.o | build
	gcc -o $(builddir)/libMCTS_ucb.so -shared $(builddir)/MCTS_ucb.o $(builddir)/game.o $(builddir)/random.o $(builddir)/dict32.o $(builddir)/aes_hash.o -pthread -lm

RMCTS.o: ./src/c/RMCTS.c ./src/c/RMCTS.h ./src/c/game.h | build
	gcc -o $(builddir)/RMCTS.o $(CFLAGS) $(include) -c ./src/c/RMCTS.c

libRMCTS.so: RMCTS.o game.o random.o | build
	gcc -o $(builddir)/libRMCTS.so -shared $(builddir)/RMCTS.o $(builddir)/game.o $(builddir)/random.o -lm
clean:
	rm -rf ./build
