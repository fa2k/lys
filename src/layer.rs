use core::fmt::Display;

struct ParameterSpec {

}

trait Layer {
    fn getparams() -> Vec<ParameterSpec>;

    fn getpar(id : u8) -> f64;
    fn setpar(id : u8, val : f64);

    fn makeframe();
}
