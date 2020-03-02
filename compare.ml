open Base
open Torch
open Bimage

let tensor_to_jpg xs ~filename =
  let grid = Tensor.view xs ~size:[ 28; 28 ] in
  let scaled_grid = Tensor.mul (Tensor.of_float0 255.) grid in
  let float_grid = Tensor.to_float2_exn scaled_grid in
  let img = Image.create u8 gray 28 28 in
  let _ = Image.for_each (fun x y _px ->
      Image.set img x y 0 (Int.of_float float_grid.(y).(x))
    ) img
  in
  Bimage_unix.Magick.write filename img

let () =
  (* Load in AE weights *)
  let vs = Var_store.create ~name:"nn" () in
  let ae = Linear.Autoencoder.create vs in
  Serialize.load_multi_ ~named_tensors:(Var_store.all_vars vs) ~filename:"weights.ot";

  (* Load in MNIST images *)
  let mnist = Mnist_helper.read_files () in

  (* Sample an image and save the before/after *)
  let images = mnist.test_images in
  let image_idx = Random.int (images |> Tensor.shape |> List.hd_exn) in
  let image = Tensor.index_select images ~dim:0 ~index:(Tensor.of_int0 image_idx) in
  let processed_image = Linear.Autoencoder.forward ae image in
  tensor_to_jpg image ~filename:"image.jpg";
  tensor_to_jpg processed_image ~filename:"processed_image.jpg";
