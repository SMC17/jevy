const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("jev_omm", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Shared library for Python ctypes/cffi
    const lib = b.addLibrary(.{
        .name = "jev_omm",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_abi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib);

    // Also install a static lib for embedding
    const lib_static = b.addLibrary(.{
        .name = "jev_omm_static",
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/c_abi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lib_static);

    // Paper demo executable
    const demo = b.addExecutable(.{
        .name = "jev_omm_demo",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/demo.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "jev_omm", .module = mod },
            },
        }),
    });
    b.installArtifact(demo);

    const demo_step = b.step("demo", "Run pure-Zig paper simulation demo");
    const demo_run = b.addRunArtifact(demo);
    if (b.args) |args| {
        demo_run.addArgs(args);
    }
    demo_step.dependOn(&demo_run.step);
    demo_run.step.dependOn(b.getInstallStep());

    // Benchmarks
    const bench = b.addExecutable(.{
        .name = "jev_omm_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "jev_omm", .module = mod },
            },
        }),
    });
    b.installArtifact(bench);

    const bench_step = b.step("bench", "Run hot-path microbenchmarks");
    const bench_run = b.addRunArtifact(bench);
    bench_step.dependOn(&bench_run.step);


    // Event-log replay harness
    const replay = b.addExecutable(.{
        .name = "jev_omm_replay",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/replay.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "jev_omm", .module = mod },
            },
        }),
    });
    b.installArtifact(replay);

    const replay_step = b.step("replay", "Replay JSONL event log → markout/PnL");
    const replay_run = b.addRunArtifact(replay);
    if (b.args) |args| {
        replay_run.addArgs(args);
    }
    replay_step.dependOn(&replay_run.step);
    replay_run.step.dependOn(b.getInstallStep());

    // Frontiers demo: SVI, multi-expiry term risk, LOB, Guéant ODE
    const frontiers = b.addExecutable(.{
        .name = "jev_omm_frontiers",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/demo_frontiers.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "jev_omm", .module = mod },
            },
        }),
    });
    b.installArtifact(frontiers);

    const frontiers_step = b.step("frontiers", "SVI, multi-expiry, LOB, and Guéant ODE paper demo");
    const frontiers_run = b.addRunArtifact(frontiers);
    frontiers_step.dependOn(&frontiers_run.step);
    frontiers_run.step.dependOn(b.getInstallStep());

    // Citadel-style training cases (paper only)
    const training = b.addExecutable(.{
        .name = "jev_omm_training",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/demo_training.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "jev_omm", .module = mod },
            },
        }),
    });
    b.installArtifact(training);

    const training_step = b.step("training", "Citadel-style paper training cases");
    const training_run = b.addRunArtifact(training);
    training_step.dependOn(&training_run.step);
    training_run.step.dependOn(b.getInstallStep());

    // Unit tests
    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    const test_step = b.step("test", "Run Zig unit tests");
    test_step.dependOn(&run_mod_tests.step);
}
