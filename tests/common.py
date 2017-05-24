import pytest
import pyopencl as cl

@pytest.fixture(scope='module')
def cl_env():
    ctx = cl.create_some_context()
    try:
        properties = cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE
        cq = cl.CommandQueue(ctx, properties=properties)
    except cl.LogicError:
        cq = cl.CommandQueue(ctx)
    return ctx, cq

