f1 = open("movielens.txt")
f2 = open("attributes.txt")

idx = 0
mapping = {}

genre_idx = 0
genre_mapping = {}
genre_count = []

og_movie_to_genre = {}

for line in f2:
    line = line.rstrip().split(",")
    line[0] = int(line[0])
    #if line[1] in ['Drama', 'Comedy', 'Western', 'Horror', 'Documentary']:
    if line[1] not in genre_mapping:
        genre_mapping[line[1]] = genre_idx
        genre_idx += 1
        genre_count.append(1)
    else:
        genre_count[genre_mapping[line[1]]] += 1

    og_movie_to_genre[line[0]] = genre_mapping[line[1]]

inrankings = []
for line in f1:
    line = line.rstrip().split(",")
    line = [int(i) for i in line]
    inrankings.append(line)

for movie in inrankings[0]:
    if movie not in mapping and movie in og_movie_to_genre:
        mapping[movie] = idx
        idx += 1

final_movie_to_genre = {}
for i in range(len(inrankings[0])):
    if inrankings[0][i] in mapping:
        mapped_idx = mapping[inrankings[0][i]]
        final_movie_to_genre[mapped_idx] = og_movie_to_genre[inrankings[0][i]]

for i in range(len(inrankings)):
    new_ranking = []
    for j in range(len(inrankings[i])):
        if inrankings[i][j] in mapping:
            new_ranking.append(mapping[inrankings[i][j]])
    inrankings[i] = new_ranking


print(genre_mapping)
print(genre_count)

print(len(inrankings), len(inrankings[0]), len(genre_mapping))

for i in range(5):
    print(0, 0)

for r in inrankings:
    print(*r)

for i in inrankings[0]:
    print(i, final_movie_to_genre[i])