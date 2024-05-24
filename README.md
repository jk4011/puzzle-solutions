# puzzle-solutions

- [X]  [Tensor-Puzzles](https://github.com/srush/Tensor-Puzzles)
- [X]  [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)
- [ ]  [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)
- [ ]  [Transformer-Puzzles](https://github.com/srush/Transformer-Puzzles)
- [ ]  [GPTWorld](https://github.com/srush/GPTWorld)
- [ ]  [Autodiff-Puzzles](https://github.com/srush/Autodiff-Puzzles)
- [ ]  [LLM-Training-Puzzles](https://github.com/srush/LLM-Training-Puzzles)
- [ ]  [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)

# 1. Lessons from Tensor-Puzzles
딥러닝 연구를 하면 다들 Numpy나 PyTorch의 메서드를 자주 사용할 것이다. 이런 매직 메서드는 추상화가 깔끔히 되어 있기 때문에 쉽게 사용할 수 있다. 하지만 우리는 이런 추상화에 숨겨져 있는 메서드의 동작 원리를 모른다. 메서드는 마치 마법을 통해 GPU에서 실행되는 것 같다. 

Tensor-Puzzles은 기본적인 행렬 연산들만을 활용해 (`+-*/, >=<, @, [:, idx]`) 매직 메서드들을 구현해 보는 퍼즐이다. 이 퍼즐을 풀어보면서 매직 메서드들이 GPU에서 어떻게 실행되는지 감을 얻을 수 있다!

# 2. Lessons from GPU-Puzzles
딥러닝 연구를 하면 GPU가 중요하다는 것은 누구나 공감할 것이다. GPU는 무거운 연산을 병렬화하여 거대한 딥러닝 모델을 훈련하고 추론하는 것을 가능하게 한다.  하지만 우리는 GPU가 어떻게 동작하는지 모른다. 우리가 아는 것은 연산이 병렬화 된다는 것뿐이다. 

GPU-Puzzles는 실제 GPU kernel을 작성하여 직접 텐서 연산을 (e.g. broadcasting, matmul) 구현해 보는 퍼즐이다. 이 퍼즐을 풀면서 GPU kernel이 어떻게 연산을 병렬화 하는지, GPU block이 어떻게 공유 메모리를 사용하는지를 배울 수 있다. 

특히 마지막 matmul은 정말 어려운데, kernel과 block level을 모두 고려해야 하기 때문이다. kernel level에서는 공유 메모리에 데이터를 올리는 cost를 최소화하는 반면, block level에서는 어떻게 연산을 독립된 작은 문제로 나눌 수 있는지를 고려해야 한다. (진짜 재밌다 ㅋㅋㅋㅋ)

