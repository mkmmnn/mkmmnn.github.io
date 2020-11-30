---
layout: post
title: Using Recurrent Neural Networks to Imagine a Collaboration Between Artists
---

## Project Idea
With the rise of music streaming software in the recent decade, it has become extremely easy to have a music taste with heavy breadth. Personally, I have noticed that the music that I listen to within a given day rarely conforms to one genre. Taking note of this, I began considering how entertaining it may be to listen to a song that was a collaboration of two artists with greatly distinct styles. My project focuses on using neural networks to create a small set of lyrics that would result from a combination of two artist's lyrical styles. 

## Framework, Neural Network Implemented
The neural networks used to train on text are genrally Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN). These neural networks (RNNs) are predictave, and generate the upcoming character based on the order of previous characters. LSTMs are a more complex version of a simple RNN. Specifically, LSTMs are great sequence predictors (order of characters within lyrics, in our case) because of their overall structure incorporates a 

## Data
I used the Genius API as the source of my lyrics. Using the API is straightforward, but it is important to control for a few possible issues. 
1. Songs with the string "(Live)" are skipped to avoid potential for redundancies. 
2. Songs with the string "(Remix)" are also skipped to avoid potential for redundancies. 
3. Once all lyrics are appended to a .txt file, they are converted into all lowercase so the number of character options for the neural network to train on will be lower. 

```python
    def makeLyricsText(artistName):
        title = artistName.replace(" ", "") + ".txt"
        file = open(title, "w")
        genius = lg.Genius('EVEbdlvJhu4oRGFz56kQIrETSGLVaJDvHkwbNzsyu1ysjU0Jc8x0w641ZqdfXmc8', skip_non_songs=True, 
                       excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
        artists = [artistName]
        
        def getLyrics(arr, k):
            c = 0
            for name in arr:
                try:
                    songs = (genius.search_artist(name, max_songs=k, sort='popularity')).songs
                    s = [song.lyrics for song in songs]
                    file.write("\n \n    \n \n".join(s))
                    c += 1
                    print(f"Songs grabbed:{len(s)}")
                except:
                    print(f"some exception at {name}: {c}")

        getLyrics(artists, 20)
        file = open(title, encoding = "UTF-8")
        specificLyrics = file.read()
        specificLyrics = specificLyrics.lower()
        file.close()
        print(specificLyrics)
        return specificLyrics
```

These lyrics were then all read and formatted into "sentences" that would be randomly fed into the neural network during training. The first step was to enumerate all possible characters that existed within the lyric set, and create a mapping mechanism for characters to incidies and vice versa. 

```python
    chars = sorted(list(set(lyrics)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
```


## Process
Initially, I made the mistake of hoping to be able to train a very robust LSTM RNN on each artist within a set of artists that I would then mix and match to create the imaginary collaborations. This did generate very accurate lyrics (few mistakes in the text, and from my qualitative point of view, quite indicative of the artist's style). I began this robust LSTM RNN on 

