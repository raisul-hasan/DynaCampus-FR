# Aborted before dataset inference

The official legacy Lightning checkpoint contained a callback class metadata key
that modern PyTorch's restricted loader rejected. No dataset image was opened,
no embedding was extracted and no result was evaluated. The compatibility fix
keeps weights-only loading and maps that one metadata key to an inert local class.
The completed experiment uses a separate timestamped directory.
