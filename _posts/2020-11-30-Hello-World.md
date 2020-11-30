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
Then, sentences were created consisting of strings of 40 characters from the lyric set. A potential shortcoming here is that a sentence has the potential to include the end of one song and the beginning of another. At this point, I was hoping to capture the artist's general vocabulary, so I hoped that this slight oversight would not strongly influence the training of the neural network. 

The above methods of data organization into sentences was consistent for all attempts at creating different neural network architectures, which will be described below. 

## Process
Overall, I made three attempts at creating neural networks to generate lyrics. The attempts and their advantages, shortcomings, and obstacles will be enumerated below in the order in which they were attempted. 

### So Many Layers!
My initial naive approach of developing a lyric generation tool was to create a model that was very robust. The first model that I created was trained on lyrics from the top 100 U2 songs. There was no issue with the data, as it was plentiful. The issue was with the architecture of the neural network. In this first iteration of lyric generation, the model I created incorporated four LSTM layers, each with 256 nodes. On top of that, there was a culminating dense layer with the softmax activation. This model took approximately five to six hours to train only one artist's lyric data. It was trained for thirty epochs, each of which used 724 sentences. I was able to generate example lyrics for this individual artist and the text had few errors (words were complete and the style generally was agreeable with that of the artist). The loss calculated for the final epochs was very low, as can be seen in the table below. 

```python
" here is no, yeah there is no end to love
we come and go
stolen days you don't give back
stolen days  "

are just enough 
 
i shouldn't be here 'cause i should be dead
i can see the lights in front of me
i believe my best days are ahead
i can see the lights in front of me
oh, jesus, if i'm still your friend
what the hell
what the hell you got for me?
i gotta get out from under my b
```

| Epoch          | Loss         | 
| :------------- | :----------: | 
| 1              | 3.0377       | 
| 2              | 3.0056       | 
| 3              | 2.8184       | 
| 4              | 2.6854       | 
| 5              | 2.5692       | 
| 6              | 2.4267       | 
| 7              | 2.2406       | 
| 8              | 1.9899       | 
| 9              | 1.6974       | 
| 10             | 1.3526       | 
| 11             | 1.0240       | 
| 12             | 0.7250       | 
| 13             | 0.4813       | 
| 14             | 0.3097       | 
| 15             | 0.1988       |
| 16             | 0.1403       |
| 17             | 0.1143       |
| 18             | 0.1024       |
| 19             | 0.1033       |
| 20             | 0.0940       |
| 21             | 0.0856       |
| 22             | 0.0869       |
| 23             | 0.0780       |
| 24             | 0.0721       |
| 25             | 0.0673       |
| 26             | 0.0686       |
| 27             | 0.0646       |
| 28             | 0.0610       |
| 29             | 0.0619       |
| 30             | 0.0615       |

I decided that this method was not practical for being able to generate lyrics from various artists, because of the time that it took to comupte. I did like how the network was very accurate with text generation and had few (if any) results, so I hoped I could hold on to the robust architecture with many LSTM layers with many nodes. 
