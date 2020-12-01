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

```
seed: 
" here is no, yeah there is no end to love
we come and go
stolen days you don't give back
stolen days  "

generated:
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

### Transfer Learning Idea
After realizing that it might take too long to create a very robust neural network for each artist individually, I decided I would attempt a transfer learning model. Abstractly, transfer learning takes a large, generalized data set, and trains a model on that data set. Then, that base model can be further differentiate when trained on a smaller data set. First, I manually selected artists that I knew were popular enough to have enough songs in Genius, and had the potential for lyrical style. The methodology for adding to this list involved scrolling through Genius' most searched artists. The final list was:
```python
artists = ["Bob Dylan", "Kanye West", "Taylor Swift", "Kendrick Lamar", "Ariana Grande", "Queen", "Migos", "Frank Ocean",
          "U2", "Neil Young", "BROCKHAMPTON", "Talking Heads", "ABBA", "Billy Joel", "Jack Johnson", "Lorde", "Adele",
          "Pitbull", "*NSYNC", "Vengaboys", "Michael Jackson"]
```
Then, the program iterated through the list and added five songs from each artist to the .txt file for the large training data set. A drawback of this method is that it is difficult to control for the length of different artists' songs. The final .txt may not have had an equal distribution of lyrics written by each artist. After pulling this data from Genius, the same pre-processing was completed. Before training a network with the same architecture as the previous example, I made some modifications. For this trial, I reduced the number of LSTM layers to one. I also decreased the number of nodes within each layer to 128. The thought process behind this was that it is not necessary for this base model to be strongly fit to the data, because I was just hoping to discover general weights that could be used on a more "overfit" model for each individual artist. 

```python
model = Sequential()
model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))
model.add(Dense(len(chars), activation='softmax'))
```

The training was also modified to include varying diversities as well as less data per epoch, but more epochs. After training, the loss at the final epoch was 1.0702, which is no where close to what was achieved with a network with more LSTM layers and more nodes. This is what would be expected, especially with using much more diverse lyrical data set. After generating this base network, I focused on trying to import the weights generated here into an identical network, re-establishing a lyric data set with over twenty songs from the specific artist, and then retraining using the more specific data. Unfortunately, this process came with the large main obstacle of the input sizes not being compatible. Using the Keras library, I was unable to determine a successful methodology to properly implement transfer learning. Though I did not achieve the proper "collaboration generation" that I was hoping for, I was able to generate some interesting lyrics that were simply an amalgamation of the words used by twenty or more artists. Some interesting generations from the base network are below:
```
seed:
"she will (timber)

swing your partner 'r"

generated:
she will (timber)

swing your partner 'round and they don't really care about us
all i wanna say is that they don't really care about us
all i wanna say is that they don't really care about us

let me not turn me (your heep), all it (hey)
give me everything tonight i had streets in the street
i knew you night
be the mind me (swerve)
it is a weepin' and then all the time
i want you to stay
i was too clowin' man
shoot (hoo side me all the
```

```
seed:
"nashin' of teeth (swerve)
it is a weepin"

generated: 
nashin' of teeth (swerve)
it is a weeping for
it's gonna be me
i'm looking for tonight

i see the scan in ohere
seess and at the cookies the same burning and the money called and tell me what i'm waitin' for
she's a mond enting nothing

i got it a walk it (walk it), walk it like i talk it (walk it), walk it like i talk it (let's go)
walk it like i talk it (walk it), walk it like i talk it (talk it), walk it like i talk it (let's go)
wal
```
The takeaways from these examples are that training on a very diverse data set results in the LSTM focusing on repitition as a main general trait of lyrics.

### The *Most* Successful Attempt
After trying the above two tactics for creating a lyric collaboration generator, I decided to attempt to use the best of both ideas I had previously tried. From the robust model, I liked how the model was trained exclusively on one artist, because I felt that it resulted in more reasonable results. From the second attempt at transfer learning, I decided to use a similar network architecture because it was able to train much faster. My plan here is to generate two different models for both inputted artists. Then, using these models, I would build a portion of a song, alternating which model was used to generate lyrics. Whichever model was used to generate lyrics would be generating lyrics based on the seed being from the alternate model. In this way, the output would be a combination of both artist's lyrical styles, and the overall "song" would be sculpted from both network's training efforts. The architecture that I used on the individual artists was the same as was used on the large data set from the transfer learning network. There was one LSTM with 128 nodes and a dense layer with softmax activation. The drawback to using a network that is not as intense as the first one is that we can expect more errors in the text production from this network. I was willing to compromise accuracy for the speed of generation so I could create more interesting collaborations between more artists. The loss from the final epoch of training was slightly better than it was for the large lyric data set, which demonstrates that it is beneficial to train a network on a single artist, because the artist's unique word choices. First, I trained a network with this architecture on Taylor Swift, and generated a loss of 0.8537. After that, I also trained a model with the same structure on Bob Dylan, and generated a loss of 1.0373. These are no where close to how accurate the network was with four LSTMs with double the number of nodes per layer. However, when generating lyrics, the number of errors in the text was not so low that it was fully meaningless. 

#### Code Structure
I created methods to:
* Generate a .txt document of lyrics of the inputted artist name string
* Format the data, create a model, and train a model on the data of the given artist
* Generate a string of 400 characters using the model returned from the above method, and an inputted seed
Then, with the power to generate lyrics from a given artist based on whatever given seed I would like, I was able to feed lyrics of one artist into the network of another, in order to generate a song that was a combination of the two. 

Code to generate lyrics.
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
Code to create the model from given lyrics
```python
def createModel(lyrics):
    chars = sorted(list(set(lyrics)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # finding num. of characters in text & number of unique characters
    n_characters = len(lyrics)
    n_vocabulary = len(chars)

    print("Total characters in lyric set: " , n_characters)
    print("Total vocabulary in lyric set: " , n_vocabulary)

    seqlen = 40
    step = seqlen
    sentences = []
    for i in range(0, len(lyrics) - seqlen - 1, step):
        sentences.append(lyrics[i: i + seqlen + 1])
    x = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), seqlen, len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, (char_in, char_out) in enumerate(zip(sentence[:-1], sentence[1:])):
            x[i, t, char_indices[char_in]] = 1
            y[i, t, char_indices[char_out]] = 1
            
    model = Sequential()
    model.add(LSTM(128, input_shape=(seqlen, len(chars)), return_sequences=True))
    model.add(Dense(len(chars), activation='softmax'))

    model.compile(
        loss = 'categorical_crossentropy',
        optimizer=RMSprop(learning_rate=0.01), 
        metrics=['categorical_crossentropy', 'accuracy'])

    def sample(preds, temperature=1.0):
        """Helper function to sample an index from a probability array."""
        preds = np.asarray(preds).astype('float64')
        preds = np.exp(np.log(preds) / temperature)  # softmax
        preds = preds / np.sum(preds)                #
        probas = np.random.multinomial(1, preds, 1)  # sample index
        return np.argmax(probas)    

    def on_epoch_end(epoch, _):
        """Function invoked at end of each epoch. Prints generated text."""
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(lyrics) - seqlen - 1)


        for diversity in [0.2, 0.5, 1.0]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = lyrics[start_index: start_index + seqlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, seqlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
                preds = model.predict(x_pred, verbose=0)
                next_index = sample(preds[0, -1], diversity)
                next_char = indices_char[next_index]

                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

    model.fit(x, y,
              batch_size=128,
              epochs=50,
              callbacks=[print_callback])
    
    return model
```
There is a final method that is used to generate the lyrics called ```generateLyricLine(artist, model, lyrics)```. I was unable to reduce the parameterization due to lack of time to refactor code. This would be a future goal of this project. Here is the code used to generate a collaboration song between Taylor Swift and Bob Dylan:
```python
def createSongCollab(artist1, artist2):
    artist1fullLyrics = makeLyricsText(artist1)
    artist2fullLyrics = makeLyricsText(artist2)
    artist1model = createModel(artist1)
    artist2model = createModel(artist2)
    artist1line1 = generateLyricLine(artist2fullLyrics[0:40], artist1, artist1fullLyrics)
    artist2line1 = generateLyricLine(artist1line1[-40:], artist2, artist2fullLyrics)
    artist1line2 = generateLyricLine(artist2line1[-40:], artist1, artist1fullLyrics)
    artist2line2 = generateLyricLine(artist1line2[-40:], artist2, artist2fullLyrics)
    song = artist1line1 + artist2line1 + artist1line2 + artist2line2
    return song

print(createSongCollab("Taylor Swift", "Bob Dylan"))
```
Using this general framework, but a different specific seed, I was able to generate a "song" to envision what a collaboration would look like between these two artists: 

>*and you'd came*
>
>*i was true*
>
>*this i did call*
>
>*that i dod't like a fay*
>
>*and you'd crustime*
>
>*i think i'vers go downthi has d dich*
>
>*to tell you'd call*
>
>*i've seen that?*
>
>*is it cool that's deain't that i to dell* my back "now you do the time the wind
>
>you that they to draging and stand in the gut stripp
>
>in is not he leany peeps
>
>they mublething to the pregiled to way
>
>they do rear
>
>the watch that he was was sta*le, i'l inded you'd call*
>
>*anv reading to teent's a that?*
>
>*is it chill that you'll seevers*
>
>*the said "james*
>
>*severy*
>
>*tantire, lover*
>
>*see this i mast's that to deaving to teat mest me do*
>
>*lough*
>
>*be babyed*
>
>*be'* my back "now you do the time the wind
>
>you that they to draging and stand in the gut stripp
>
>in is not he leany peeps
>
>they mublething to the pregiled to way
>
>they do rear
>
>the watch that he was was sta

Italicized lyrics are from Taylor Swift's neural network, and regular text is from Bob Dylan. It is somewhat difficult to distinguish between their lyrical styles in the above song, and there are numerous text errors (this is most likely because the network was not as robust as it could have been). 


I also was able to train another network on the musical group, Migos, and was subsequently able to see what a collaboration between Migos and Bob Dylan could look like. These two artists most likely would have very different vocabularies, or at least much more distinct than that of Bob Dylan and Taylor Swift. Below is the resulting generation:


