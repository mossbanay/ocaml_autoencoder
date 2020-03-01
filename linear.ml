open Base
open Torch

(* One hidden layer with 128 nodes and embedding dimension of 20 *)
let hidden_nodes = [ 128; 20 ]
let num_epochs = 2000
let learning_rate = 1e-3
let batch_size = 256

let () =
  let device = Device.cuda_if_available () in
  let mnist = Mnist_helper.read_files () in
  let vs = Var_store.create ~name:"nn" () in

  let layer_shapes =
    let layer_dims = List.cons Mnist_helper.image_dim hidden_nodes in
    let in_dims =
      List.append layer_dims (layer_dims |> List.tl_exn |> List.rev |> List.tl_exn)
    in
    let out_dims = List.rev in_dims in
    List.fold2_exn in_dims out_dims
      ~init:[] ~f:(fun acc in_dim out_dim -> List.cons (in_dim, out_dim) acc)
    |> List.rev
  in

  let layers = List.map
      ~f:(fun (in_dim, out_dim) -> Layer.linear vs out_dim ~activation:Relu ~input_dim:in_dim)
      layer_shapes
  in

  let adam = Optimizer.adam vs ~learning_rate in

  let model xs =
    List.fold ~init:xs ~f:(fun l x -> Layer.forward x l) layers
  in

  for batch_idx = 1 to num_epochs do
    let batch_images, _ = Dataset_helper.train_batch mnist ~device ~batch_size ~batch_idx in

    let loss = Tensor.mse_loss (model batch_images) batch_images in
    Optimizer.backward_step adam ~loss;

    (* Print updates every 50 epochs *)
    if batch_idx % 50 = 0
    then (
      let train_mse =
        Tensor.mse_loss (model mnist.train_images) mnist.train_images
        |> Tensor.float_value
      in

      let test_mse =
        Tensor.mse_loss (model mnist.test_images) mnist.test_images
        |> Tensor.float_value
      in

      Stdio.printf
        "Epoch: %4d, Training MSE: %.4f Test MSE: %.4f\n%!"
        batch_idx
        (100. *. train_mse)
        (100. *. test_mse);
    );

    Caml.Gc.full_major ()
  done
