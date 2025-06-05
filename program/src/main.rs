// These two lines are necessary for the program to properly compile.
//
// Under the hood, we wrap your main function with some extra code so that it behaves properly
// inside the zkVM.
#![no_main]
sp1_zkvm::entrypoint!(main);
use appleproof_lib::has_apple;

pub fn main() {
    let image_vec = sp1_zkvm::io::read_vec();
    let is_apple = has_apple(image_vec);
    let apple_slice = if is_apple { vec![1u8] } else { vec![0u8] };
    sp1_zkvm::io::commit_slice(apple_slice.as_slice());
}
