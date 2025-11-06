enum DeviceType {
    Cpu,
    Gpu,
    Any,
}

pub trait DeviceInfo {
    fn device() -> DeviceType;
}

pub struct GPU;
pub struct CPU;
pub struct AnyDevice;

impl DeviceInfo for CPU {
    #[inline]
    fn device() -> DeviceType {
        DeviceType::Cpu
    }
}

impl DeviceInfo for GPU {
    #[inline]
    fn device() -> DeviceType {
        DeviceType::Gpu
    }
}

impl DeviceInfo for AnyDevice {
    #[inline]
    fn device() -> DeviceType {
        DeviceType::Any
    }
}
