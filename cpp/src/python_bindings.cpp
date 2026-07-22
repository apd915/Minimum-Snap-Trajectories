#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "trajectory_planner.hpp"
#include "front_end.hpp"
#include "voxel_grid.hpp"

namespace py = pybind11;
using namespace trajectory_planner;

PYBIND11_MODULE(min_snap_bindings, m) {
    m.doc() = "Python bindings for the C++ Minimum Snap Trajectory Planner";

    py::class_<mapping::ObstacleBox, std::unique_ptr<mapping::ObstacleBox>>(m, "ObstacleBox")
        .def(py::init([](const std::vector<double>& min, const std::vector<double>& max) {
            return std::make_unique<mapping::ObstacleBox>(mapping::ObstacleBox{
                Eigen::Vector3d(min[0], min[1], min[2]),
                Eigen::Vector3d(max[0], max[1], max[2])
            });
        }))
        .def_property("min_pt", 
            [](const mapping::ObstacleBox& b) { return std::vector<double>{b.min.x(), b.min.y(), b.min.z()}; },
            [](mapping::ObstacleBox& b, const std::vector<double>& v) { b.min = Eigen::Vector3d(v[0], v[1], v[2]); })
        .def_property("max_pt", 
            [](const mapping::ObstacleBox& b) { return std::vector<double>{b.max.x(), b.max.y(), b.max.z()}; },
            [](mapping::ObstacleBox& b, const std::vector<double>& v) { b.max = Eigen::Vector3d(v[0], v[1], v[2]); });

    py::class_<FrontEndConfig, std::unique_ptr<FrontEndConfig>>(m, "FrontEndConfig")
        .def(py::init<>())
        .def_readwrite("voxel_resolution", &FrontEndConfig::voxel_resolution)
        .def_readwrite("drone_physical_radius", &FrontEndConfig::drone_physical_radius)
        .def_readwrite("sfc_height", &FrontEndConfig::sfc_height)
        .def_readwrite("sfc_width", &FrontEndConfig::sfc_width)
        .def_readwrite("sfc_start_ext", &FrontEndConfig::sfc_start_ext)
        .def_readwrite("sfc_end_ext", &FrontEndConfig::sfc_end_ext)
        .def_readwrite("spline_type", &FrontEndConfig::spline_type)
        .def_readwrite("aircraft_type", &FrontEndConfig::aircraft_type)
        .def_readwrite("degree", &FrontEndConfig::degree)
        .def_property("map_bounds",
            [](const FrontEndConfig& c) { return std::vector<double>{c.map_bounds.x(), c.map_bounds.y(), c.map_bounds.z()}; },
            [](FrontEndConfig& c, const std::vector<double>& v) { c.map_bounds = Eigen::Vector3d(v[0], v[1], v[2]); });

    py::class_<PlanningResult, std::unique_ptr<PlanningResult>>(m, "PlanningResult")
        .def_readonly("control_points", &PlanningResult::control_points)
        .def_readonly("total_control_points", &PlanningResult::total_control_points)
        .def_readonly("success", &PlanningResult::success)
        .def_readonly("sfc_time_ms", &PlanningResult::sfc_time_ms)
        .def_readonly("opt_time_ms", &PlanningResult::opt_time_ms)
        .def_readonly("overhead_ms", &PlanningResult::overhead_ms)
        .def_readonly("total_time_ms", &PlanningResult::total_time_ms);

    py::class_<TrajectoryPlanner, std::unique_ptr<TrajectoryPlanner>>(m, "TrajectoryPlanner")
        .def(py::init<const FrontEndConfig&, const std::vector<mapping::ObstacleBox>&, double, double>(),
             py::arg("config"), py::arg("obstacles"), py::arg("v_max") = 3.0, py::arg("a_max") = 2.0)
        .def("plan_mission", [](TrajectoryPlanner& planner, const std::vector<double>& start, const std::vector<double>& goal) {
            return planner.plan_mission(Eigen::Vector3d(start[0], start[1], start[2]), Eigen::Vector3d(goal[0], goal[1], goal[2]));
        });
}
