You task is to implement an simple interpreter for the notorious esoteric language [HQ9+](https://esolangs.org/wiki/HQ9+) that will work for a single character input:

- If the input is `'H'`, return `'Hello World!'`
- If the input is `'Q'`, return the input
- If the input is `'9'`, return the full lyrics of [99 Bottles of Beer](http://www.99-bottles-of-beer.net/lyrics.html). It should be formatted like this:

```if:rust
__Note__: In Rust, return `Some` containing the appropriate value.
```

```
99 bottles of beer on the wall, 99 bottles of beer.
Take one down and pass it around, 98 bottles of beer on the wall.
98 bottles of beer on the wall, 98 bottles of beer.
Take one down and pass it around, 97 bottles of beer on the wall.
97 bottles of beer on the wall, 97 bottles of beer.
Take one down and pass it around, 96 bottles of beer on the wall.
...
...
...
2 bottles of beer on the wall, 2 bottles of beer.
Take one down and pass it around, 1 bottle of beer on the wall.
1 bottle of beer on the wall, 1 bottle of beer.
Take one down and pass it around, no more bottles of beer on the wall.
No more bottles of beer on the wall, no more bottles of beer.
Go to the store and buy some more, 99 bottles of beer on the wall.
```

- For everything else, don't return anything (return `null` in C#, `None` in Rust).

(`+` has no visible effects so we can safely ignore it.)