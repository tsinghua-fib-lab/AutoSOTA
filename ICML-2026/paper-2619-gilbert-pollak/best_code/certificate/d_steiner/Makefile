all: plot_f_ge_d plot_f_le_d

%: %.cpp
	g++ $< -O3 -pthread -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -std=c++23 -ffast-math -o $@
