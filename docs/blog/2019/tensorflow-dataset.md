When I saw the TensorFlow Dev Summit 2019, the thing that I wanted to try out the most was the new [`tf.data.Dataset API`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset). We all know how painful it is to feed data to our models in an efficient way. This is especially true if you're working with Speech.

For my work with [Pointer-Networks](https://arxiv.org/abs/1506.03134), I was using PyTorch's DataLoader to feed data to my models. This always left something to be desired (a discussion for another day). I was on the lookout for a different (hopefully better) way to feed data to my models when I heard about tf.data.Dataset.

This post is my exploration of the API, I will try to keep this post updated as I go about my exploration.
I am exploring the following APIs: [tf.data.Dataset.from_generator](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#from_generator), [tf.data.Options](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Options), [tf.data.TFRecordDataset](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/TFRecordDataset) and all the other experimental features!


## tf.data.Dataset.from_generator

[tf.data.Dataset.from_generator](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#from_generator) is the function to use if your data pipeline does not fit into any of the other methods.
Since I mostly work with speech, I need a way to load my data from disk batch-by-batch. I can't fit all my data into memory because it's just too big (typically couple of `100 GiBs`).

One way to feed such dataset to my models is by loading the data batch-by-batch from the disk instead of loading everything at once and iterating over it. This has always been one of the most difficult part of my model building experience in Speech. Having an efficient data pipeline makes my life easier :).

[ESPnet](https://github.com/espnet/espnet) did just that by using the [ark file splits generated by kaldi](http://kaldi-asr.org/doc/io.html) to load the batches and feed them to my models. This is definitely not THE solution to the problem, but it got the job done.

I believe the tf.data.Dataset.from_generator is the way to go for my data pipeline.

Now, let's say I need to solve the problem of finding [ConvexHull](https://en.wikipedia.org/wiki/Convex_hull) points from a sequence of points. This is one of the problems the original Pointer-Networks paper tried to solve. Instead of using the dataset that the authors provided, I want to generate my own dataset (because why not? how difficult could it be to generate a set of points to solve this problem?). By generating my own dataset, I can practically have infinite training examples and full control over what I want to do with it.

For this reason alone, I can't use the other methods as I will have to store the training examples in memory. I need to generate my examples on-the-go.

`tf.data.Dataset.from_generator` solves this exact problem.

### How to use it?

Before we even start feeding data to our model, we need to have a python [generator](https://www.programiz.com/python-programming/generator) function which generates **one** training pair needed for our model.

What this means is, there should be a function which has a **`yield`** statement instead of a **`return`** statement. This does not mean there can't be a return statement, in a generator function there could be multiple yields and returns.

Let's say our dataset is of `1000` images of size `28x28` and belong to one of `10` classes. Our generator function might look something like this except we will be reading the images from disk.


``` py
def our_generator():
    for i in range(1000):
      x = np.random.rand(28,28)
      y = np.random.randint(1,10, size=1)
      yield x,y
```

We could build our `TensorFlow` dataset with this generator function.
The `tf.data.Dataset.from_generator` function has the following arguments:

``` py
def from_generator(
    generator,
    output_types,
    output_shapes=None,
    args=None
)
```

While the **`output_shapes`** is optional, we need to specify the output_types. In this particular case the first returned value is a 2D array of floats and the second value is a 1D array of integers. Our dataset object will look something like this:

``` py
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
```

To use this dataset in our model training, we need to either use the [**`make_one_shot_iterator`**](https://www.tensorflow.org/api_docs/python/tf/data/make_one_shot_iterator) (which is being deprecated) or use the dataset in our training loop.

#### Using make_one_shot_iterator

``` py
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
iterator = dataset.make_one_shot_iterator()
x,y = iterator.get_next()
print(x.shape, y.shape)

#(28, 28) (1,)
```

#### Loop over the dataset object in our training loop

``` py
for batch, (x,y) in enumerate(dataset):
  pass
print("batch: ", epoch)
print("Data shape: ", x.shape, y.shape)

#batch:  999
#Data shape:  (28, 28) (1,)
```

### tf.data.Dataset options - batch, repeat, shuffle

tf.data.Dataset comes with a couple of options to make our lives easier. If you see our previous example, we get one example every time we call the dataset object. What if we would want a batch of examples, or if we want to iterate over the dataset many times, or if we want to shuffle the dataset after every epoch.

Using the batch, repeat, and shuffle function we could achieve this.

``` py
def our_generator():
    for i in range(1000):
      x = np.random.rand(28,28)
      y = np.random.randint(1,10, size=1)
      yield x,y
    
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
```
#### batch

``` py
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.batch(10)
iterator = dataset.make_one_shot_iterator()
x,y = iterator.get_next()
print(x.shape, y.shape)

#(10, 28, 28) (10, 1)
```

Now, every time we use the dataset object, the generator function is called 10 times. The batch function combines consecutive elements of this dataset into batches.
If we reach the end of the dataset and the batch is less than the batch_size specified, we can pass the argument **`drop_remainder=True`** to ignore that particular batch.


``` py
for batch, (x,y) in enumerate(dataset):
  pass
print("batch: ", epoch)
print("Data shape: ", x.shape, y.shape)

#batch:  99
#Data shape:  (10, 28, 28) (10, 1)
```

#### repeat


``` py
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.batch(batch_size=10)
dataset = dataset.repeat(count=2)

for batch, (x,y) in enumerate(dataset):
  pass
print("batch: ", batch)
print("Data shape: ", x.shape, y.shape)

#batch:  199
#Data shape:  (10, 28, 28) (10, 1)
```
Here, the dataset is looped over 2 times. Hence we get twice the number of batches for training. If we want to repeat the dataset indefinitely, we should set the argument to **count=-1**

#### shuffle

``` py
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.batch(batch_size=10)
dataset = dataset.repeat(count=2)
dataset = dataset.shuffle(buffer_size=1000)
iterator = dataset.make_one_shot_iterator()

for batch, (x,y) in enumerate(dataset):
  pass
print("batch: ", batch)
print("Data shape: ", x.shape, y.shape)

#batch:  199
#Data shape:  (10, 28, 28) (10, 1)
```

Here, the argument **buffer_size=100** specifies the number of elements from this dataset from which the new dataset will sample. Essentially, this fills the dataset with **buffer_size** elements, then randomly samples elements from this buffer.

Use **buffer_size>=dataset_size** for perfect shuffling.

#### Other options

In addition to batch, repeat, and shuffle, there are many other functions the `TensorFlow Dataset` API comes with. I will update this post with options like - `map`, `reduce`, `with_options`


## Conclusion

`tf.data.Dataset` potentially can solve most of my data pipeline woes. I will test how I can use this to feed speech data (use a py_function to do feature extraction) to my models, and using the map function to augment the dataset (adding noise, combining files, time scaling etc).

You can use this notebook to play around with the functions that I have used.
[https://colab.research.google.com/drive/1XxHNtgwFVZzILlOwEhvsYuov5s1MAy2N](https://colab.research.google.com/drive/1XxHNtgwFVZzILlOwEhvsYuov5s1MAy2N)