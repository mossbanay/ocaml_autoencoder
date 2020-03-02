open Base
open Torch
open Linear

(* Default training parameters *)
let num_epochs = 2000
let learning_rate = 1e-3
let batch_size = 256
let weights_filename = "weights.ot"

let () =
  let device = Device.cuda_if_available () in
  let mnist = Mnist_helper.read_files () in
  let vs = Var_store.create ~name:"nn" () in
  let ae = Autoencoder.create vs in
  let adam = Optimizer.adam vs ~learning_rate in

  for batch_idx = 1 to num_epochs do
    let batch_images, _ = Dataset_helper.train_batch mnist ~device ~batch_size ~batch_idx in

    let loss = Tensor.mse_loss (Autoencoder.forward ae batch_images) batch_images in
    Optimizer.backward_step adam ~loss;

    (* Print updates every 50 epochs *)
    if batch_idx % 50 = 0
    then (
      let train_mse =
        Tensor.mse_loss (Autoencoder.forward ae mnist.train_images) mnist.train_images
        |> Tensor.float_value
      in

      let test_mse =
        Tensor.mse_loss (Autoencoder.forward ae mnist.test_images) mnist.test_images
        |> Tensor.float_value
      in

      Stdio.printf
        "Epoch: %4d, Training MSE: %.4f Test MSE: %.4f\n%!"
        batch_idx
        (100. *. train_mse)
        (100. *. test_mse);
    );

    Caml.Gc.full_major ()
  done;

  Serialize.save_multi ~named_tensors:(Var_store.all_vars vs) ~filename:weights_filename;
