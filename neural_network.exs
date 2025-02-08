
Mix.install([
  {:nx, "~> 0.5"}
])

defmodule NeuralNetwork do
  def trying() do
    n = {2, 3, 3, 1}
    key = Nx.Random.key(1)


    {w1, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 1), elem(n, 0)})
    {w2, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 2), elem(n, 1)})
    {w3, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 3), elem(n, 2)})


    {b1, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 1), 1})
    {b2, key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 2), 1})
    {b3, _key} = Nx.Random.normal(key, 0, 1, shape: {elem(n, 3), 1})

    {a0, y, m} = prepare_data()

    epochs = 1000
    alpha = 0.1
    costs = []


    {w1, w2, w3, b1, b2, b3, costs} = Enum.reduce(0..epochs, {w1, w2, w3, b1, b2, b3, costs},
      fn epoch, {w1, w2, w3, b1, b2, b3, costs} ->

        {y_hat, cache} = feed_forward(w1, w2, w3, b1, b2, b3, a0)


        error = cost(y_hat, y)


        {dC_dW3, dC_db3, dC_dA2} = backprop_layer_3(y_hat, y, m, cache["a2"], w3)
        {dC_dW2, dC_db2, dC_dA1} = backprop_layer_2(dC_dA2, cache["a1"], cache["a2"], w2, m)
        {dC_dW1, dC_db1} = backprop_layer_1(dC_dA1, cache["a1"], a0, w1, m)


        w3 = Nx.subtract(w3, Nx.multiply(alpha, dC_dW3))
        w2 = Nx.subtract(w2, Nx.multiply(alpha, dC_dW2))
        w1 = Nx.subtract(w1, Nx.multiply(alpha, dC_dW1))
        b3 = Nx.subtract(b3, Nx.multiply(alpha, dC_db3))
        b2 = Nx.subtract(b2, Nx.multiply(alpha, dC_db2))
        b1 = Nx.subtract(b1, Nx.multiply(alpha, dC_db1))


        if rem(epoch, 20) == 0 do
          IO.puts("epoch #{epoch}: cost = #{Nx.to_number(error)}")
        end

        {w1, w2, w3, b1, b2, b3, [error | costs]}
    end)

    Enum.reverse(costs)
  end

  def feed_forward(w1, w2, w3, b1, b2, b3, a0) do

    z1 = Nx.dot(w1, a0)
    z1 = Nx.broadcast(b1, {elem(Nx.shape(z1), 0), elem(Nx.shape(z1), 1)}) |> Nx.add(z1)
    a1 = sigmoid(z1)


    z2 = Nx.dot(w2, a1)
    z2 = Nx.broadcast(b2, {elem(Nx.shape(z2), 0), elem(Nx.shape(z2), 1)}) |> Nx.add(z2)
    a2 = sigmoid(z2)


    z3 = Nx.dot(w3, a2)
    z3 = Nx.broadcast(b3, {elem(Nx.shape(z3), 0), elem(Nx.shape(z3), 1)}) |> Nx.add(z3)
    y_hat = sigmoid(z3)

    cache = %{
      "a0" => a0,
      "a1" => a1,
      "a2" => a2
    }

    {y_hat, cache}
  end

  def prepare_data() do
    x = Nx.tensor([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
    ])

    m = 10
    y = Nx.tensor([0,1,1,0,0,1,1,0,1,0])
    y = Nx.reshape(y, {1, m})

    a0 = Nx.transpose(x)  # 2x10
    {a0, y, m}
  end

  def sigmoid(z) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.multiply(z, -1))))
  end

  def cost(y_hat, y) do
    y_hat = Nx.clip(y_hat, 1.0e-7, 1 - 1.0e-7)
    losses = Nx.multiply(-1, Nx.add(
      Nx.multiply(y, Nx.log(y_hat)),
      Nx.multiply(Nx.subtract(1, y), Nx.log(Nx.subtract(1, y_hat)))
    ))

    m = elem(Nx.shape(Nx.flatten(y_hat)), 0)
    summed_losses = Nx.divide(Nx.sum(losses, axes: [1]), m)
    Nx.sum(summed_losses)
  end

  def backprop_layer_3(y_hat, y, m, a2, w3) do

    dC_dZ3 = Nx.subtract(y_hat, y) |> Nx.divide(m)

    dC_dW3 = Nx.dot(dC_dZ3, Nx.transpose(a2))

    dC_db3 = Nx.sum(dC_dZ3, axes: [1], keep_axes: true)

    dC_dA2 = Nx.dot(Nx.transpose(w3), dC_dZ3)

    {dC_dW3, dC_db3, dC_dA2}
  end

  def backprop_layer_2(propagator_dC_dA2, a1, a2, w2, m) do

    dA2_dZ2 = Nx.multiply(a2, Nx.subtract(1, a2))
    dC_dZ2 = Nx.multiply(propagator_dC_dA2, dA2_dZ2)


    dC_dW2 = Nx.dot(dC_dZ2, Nx.transpose(a1))


    dC_db2 = Nx.sum(dC_dZ2, axes: [1], keep_axes: true)

    dC_dA1 = Nx.dot(Nx.transpose(w2), dC_dZ2)

    {dC_dW2, dC_db2, dC_dA1}
  end

  def backprop_layer_1(propagator_dC_dA1, a1, a0, w1, m) do

    dA1_dZ1 = Nx.multiply(a1, Nx.subtract(1, a1))
    dC_dZ1 = Nx.multiply(propagator_dC_dA1, dA1_dZ1)


    dC_dW1 = Nx.dot(dC_dZ1, Nx.transpose(a0))


    dC_db1 = Nx.sum(dC_dZ1, axes: [1], keep_axes: true)

    {dC_dW1, dC_db1}
  end
end
NeuralNetwork.trying()
