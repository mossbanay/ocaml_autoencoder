open Base
open Torch

module Autoencoder = struct
  type t =
    { encode_layers : Layer.t list
    ; decode_layers : Layer.t list
    }

  (* One hidden layer with 128 nodes and embedding dimension of 20 by default *)
  let create ?(hidden_nodes=128) ?(embedding_size=20) vs =
    { encode_layers = [ Layer.linear vs ~activation:Relu ~input_dim:784 hidden_nodes
                      ; Layer.linear vs ~activation:Relu ~input_dim:hidden_nodes embedding_size
                      ]
    ; decode_layers = [ Layer.linear vs ~activation:Relu ~input_dim:embedding_size hidden_nodes
                      ; Layer.linear vs ~activation:Relu ~input_dim:hidden_nodes 784
                      ]
    }

  let encode t xs =
    List.fold ~init:xs ~f:(fun l x -> Layer.forward x l) t.encode_layers

  let decode t zs =
    List.fold ~init:zs ~f:(fun l x -> Layer.forward x l) t.decode_layers

  let forward t xs =
    decode t (encode t xs)
end
