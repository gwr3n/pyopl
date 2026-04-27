import os
import tempfile
import unittest

from pyopl.pyopl_core import (
    GurobiCodeGenerator,
    OPLCompiler,
    OPLLexer,
    OPLParser,
    load_opl_model,
    solve,
)


def setUpModule():
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("pyopl.scipy_codegen_csc").setLevel(logging.DEBUG)
    logging.getLogger("pyopl.gurobi_codegen").setLevel(logging.DEBUG)


# Import pyopl interface
try:
    import pyopl

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


class TestPyOPLProblems(unittest.TestCase):
    @unittest.skip("this test is cumbersome to run")
    def test_hotel_rostering(self):
        """
        Test the hotel rostering problem, a realistic and moderately complex MILP with 12 employees, 33 shifts, and 3 days.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                // Hotel rota / staff scheduling MILP
                //
                // Builds a hotel rota from preprocessed employee, availability/preference, and shift-requirement CSV data.
                // The model assigns employees to required shift records, allows penalized unfilled demand, respects
                // role/skill eligibility, availability windows, approved hard absences, night/weekend permissions,
                // rest conflicts, and balances coverage against preferences and fairness targets.

                // -------------------- Sets and index ranges --------------------
                int NumEmployees = ...;          // number of employees from employees.csv
                range Employees = 1..NumEmployees;

                int NumDays = ...;               // number of rota days in the planning instance
                range Days = 1..NumDays;

                int NumShifts = ...;             // number of required shift records from shift_requirements.csv
                range Shifts = 1..NumShifts;

                // -------------------- Employee parameters --------------------
                param float contractHours[Employees] = ...;       // contractual weekly hours by employee
                param float maxWeeklyHours[Employees] = ...;      // maximum allowed weekly hours by employee
                param float maxDailyHours[Employees] = ...;       // daily paid-hour cap by employee
                param float minRestHours[Employees] = ...;        // required rest hours between non-overlapping worked shifts
                param float targetHours[Employees] = ...;         // fair target hours for this rota horizon

                // Historical fairness metrics used to discourage repeatedly assigning unpopular shift types.
                param int histWeekend[Employees] = ...;           // weekend shifts in previous 4 weeks
                param int histLate[Employees] = ...;              // late shifts in previous 4 weeks
                param int histNight[Employees] = ...;             // night shifts in previous 4 weeks

                // Soft day-off penalty. Only soft requested-off rows remain assignable; hard absences are blocked below.
                param float dayOffPenalty[Employees] = ...;       // penalty for assigning an approved soft requested day off

                // -------------------- Shift parameters --------------------
                param int dayIndex[Shifts] = ...;                 // rota day of each shift, in Days
                param float shiftHours[Shifts] = ...;             // paid hours for one assignment to each shift
                param float shiftStartAbs[Shifts] = ...;          // shift start in absolute hours from start of horizon
                param float shiftEndAbs[Shifts] = ...;            // shift end in absolute hours; overnight shifts end next day
                param int requiredStaff[Shifts] = ...;            // required number of employees on each shift
                param float priorityPenalty[Shifts] = ...;        // dominant penalty per unfilled staff slot by priority

                param int isWeekendShift[Shifts] = ...;           // 1 if shift is on a weekend day, else 0
                param int isLateShift[Shifts] = ...;              // 1 if shift is late/evening, else 0
                param int isNightShift[Shifts] = ...;             // 1 if shift is overnight, else 0

                // -------------------- Preprocessed employee-shift matrices --------------------
                param int roleEligible[Employees][Shifts] = ...;       // 1 if role, transferable skill, night, and weekend permissions allow cover
                param int windowOK[Employees][Shifts] = ...;           // 1 if shift lies within an allowed or approved exception time window
                param int hardUnavailable[Employees][Shifts] = ...;    // 1 if employee has a hard absence for the shift date; assignment forbidden
                param int prefMatch[Employees][Shifts] = ...;          // 1 if shift type matches employee/request preference or approved exception
                param int requestedOff[Employees][Shifts] = ...;       // 1 if the shift falls on a requested day off

                // Combined hard eligibility: role/skill permission, time-window feasibility, and no hard absence.
                param int eligible[e in Employees][s in Shifts] =
                roleEligible[e][s] * windowOK[e][s] * (1 - hardUnavailable[e][s]);

                // Same-employee shift conflict: overlap or insufficient rest, except for the documented
                // housekeeping supervisor concurrent HK_AM/HK_SUP cover on each day.
                param boolean shiftConflict[e in Employees][s in Shifts][t in Shifts] =
                (s < t)
                && !((e == 5) && (((s == 4) && (t == 5)) || ((s == 15) && (t == 16)) || ((s == 26) && (t == 27))))
                && !((shiftStartAbs[t] >= shiftEndAbs[s] + minRestHours[e]) || (shiftStartAbs[s] >= shiftEndAbs[t] + minRestHours[e]));

                // -------------------- Objective weights --------------------
                param float prefMismatchPenalty = ...;            // penalty for assigning a non-preferred shift type
                param float underTargetPenalty = ...;             // penalty per hour below targetHours
                param float overTargetPenalty = ...;              // penalty per hour above targetHours
                param float lateFairnessPenalty = ...;            // marginal penalty scaled by historical late count
                param float nightFairnessPenalty = ...;           // marginal penalty scaled by historical night count
                param float weekendFairnessPenalty = ...;         // marginal penalty scaled by historical weekend count

                // -------------------- Decision variables --------------------
                dvar boolean x[Employees][Shifts];                // 1 if employee e is assigned to shift record s

                dvar int+ unfilled[Shifts];                       // shortage against requiredStaff[s]; keeps rota feasible

                dvar float+ underTarget[Employees];               // hours below the fair target over this rota horizon
                dvar float+ overTarget[Employees];                // hours above the fair target over this rota horizon

                // Assigned paid hours per employee over the rota horizon.
                dexpr float assignedHours[e in Employees] =
                sum(s in Shifts) (shiftHours[s] * x[e][s]);

                // -------------------- Objective --------------------
                // TotalWeightedPenalty: shortage penalties are set dominantly in data so feasible coverage is preferred
                // before soft day-off use, preference mismatch, target-hour deviation, and historical fairness costs.
                minimize TotalWeightedPenalty =
                    sum(s in Shifts) (priorityPenalty[s] * unfilled[s])
                + sum(e in Employees, s in Shifts) (dayOffPenalty[e] * requestedOff[e][s] * x[e][s])
                + sum(e in Employees, s in Shifts) (prefMismatchPenalty * (1 - prefMatch[e][s]) * x[e][s])
                + sum(e in Employees) (underTargetPenalty * underTarget[e] + overTargetPenalty * overTarget[e])
                + sum(e in Employees, s in Shifts)
                    ((lateFairnessPenalty * histLate[e] * isLateShift[s]
                    + nightFairnessPenalty * histNight[e] * isNightShift[s]
                    + weekendFairnessPenalty * histWeekend[e] * isWeekendShift[s]) * x[e][s]);

                // -------------------- Constraints --------------------
                subject to {
                // CoverRequired: each shift requirement is covered by assigned staff plus explicit shortage.
                forall(s in Shifts)
                    CoverRequired:
                    sum(e in Employees) x[e][s] + unfilled[s] == requiredStaff[s];

                // Eligibility: assignments must respect role/skill, permissions, time windows, and hard absences.
                forall(e in Employees, s in Shifts)
                    Eligibility:
                    x[e][s] <= eligible[e][s];

                // DailyHoursCap: keep each employee's total paid hours within the operational daily limit.
                forall(e in Employees, d in Days)
                    DailyHoursCap:
                    sum(s in Shifts : dayIndex[s] == d) (shiftHours[s] * x[e][s]) <= maxDailyHours[e];

                // WeeklyHoursCap: keep each employee's total rota hours within their maximum weekly hours.
                forall(e in Employees)
                    WeeklyHoursCap:
                    assignedHours[e] <= maxWeeklyHours[e];

                // TargetHoursBalance: define soft under/over deviations around fair horizon target hours.
                forall(e in Employees)
                    TargetHoursBalance:
                    assignedHours[e] + underTarget[e] - overTarget[e] == targetHours[e];

                // RestAndOverlap: prevent overlapping or too-close shifts, except preprocessed permitted concurrent cover.
                forall(e in Employees, s in Shifts, t in Shifts : shiftConflict[e][s][t])
                    RestAndOverlap:
                    x[e][s] + x[e][t] <= 1;
                }
                """
            data_code = """
                NumEmployees = 12;
                NumDays = 3;
                NumShifts = 33;

                // Employee order:
                // 1 E001 Amy Fraser; 2 E002 Callum Reid; 3 E003 Sophie Murray; 4 E004 Leah Kerr;
                // 5 E005 Euan McBride; 6 E006 Mina Ali; 7 E007 Daniel Ross; 8 E008 Holly Campbell;
                // 9 E009 Jamie Stewart; 10 E010 Rory Grant; 11 E011 Isla Robertson; 12 E012 Noah Sinclair.

                // Shift order by index:
                // 1 D1_FO_EARLY; 2 D1_FO_LATE; 3 D1_FO_NIGHT; 4 D1_HK_AM; 5 D1_HK_SUP; 6 D1_FB_BREAKFAST; 7 D1_FB_LATE; 8 D1_FB_BAR; 9 D1_KIT_EARLY; 10 D1_KIT_LATE; 11 D1_MT_DAY;
                // 12 D2_FO_EARLY; 13 D2_FO_LATE; 14 D2_FO_NIGHT; 15 D2_HK_AM; 16 D2_HK_SUP; 17 D2_FB_BREAKFAST; 18 D2_FB_LATE; 19 D2_FB_BAR; 20 D2_KIT_EARLY; 21 D2_KIT_LATE; 22 D2_MT_DAY;
                // 23 D3_FO_EARLY; 24 D3_FO_LATE; 25 D3_FO_NIGHT; 26 D3_HK_AM; 27 D3_HK_SUP; 28 D3_FB_BREAKFAST; 29 D3_FB_LATE; 30 D3_FB_BAR; 31 D3_KIT_EARLY; 32 D3_KIT_LATE; 33 D3_MT_DAY.

                contractHours = [40, 24, 40, 20, 37.5, 16, 40, 45, 24, 40, 20, 20];
                maxWeeklyHours = [48, 30, 48, 28, 45, 24, 48, 52, 32, 45, 25, 25];
                maxDailyHours = [12, 12, 12, 12, 14, 12, 12, 12, 12, 12, 12, 12];
                minRestHours = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11];

                // Three-day fair-hour targets derived from weekly contract hours.
                targetHours = [17.14, 10.29, 17.14, 8.57, 16.07, 6.86, 17.14, 19.29, 10.29, 17.14, 8.57, 8.57];

                histWeekend = [2, 1, 1, 2, 3, 1, 2, 1, 1, 0, 0, 1];
                histLate = [1, 4, 0, 0, 0, 2, 3, 0, 2, 0, 0, 0];
                histNight = [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0];

                // E011 has a lower soft day-off penalty for the approved D3 FO_EARLY cover exception.
                dayOffPenalty = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 400, 1000];

                dayIndex = [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
                ];

                shiftHours = [
                8, 8, 8, 6, 8, 6, 6, 8, 8, 8, 8,
                8, 8, 8, 6, 8, 6, 6, 8, 8, 8, 8,
                8, 8, 8, 6, 8, 6, 6, 8, 8, 8, 8
                ];

                // Absolute hours from 2024-07-01 00:00; overnight shifts end after midnight.
                shiftStartAbs = [
                7, 15, 23, 8, 8, 6.5, 17, 16, 6, 15, 9,
                31, 39, 47, 32, 32, 30.5, 41, 40, 30, 39, 33,
                55, 63, 71, 56, 56, 54.5, 65, 64, 54, 63, 57
                ];
                shiftEndAbs = [
                15, 23, 31, 14, 16, 12.5, 23, 24, 14, 23, 17,
                39, 47, 55, 38, 40, 36.5, 47, 48, 38, 47, 41,
                63, 71, 79, 62, 64, 60.5, 71, 72, 62, 71, 65
                ];

                requiredStaff = [
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1
                ];

                // Dominant shortage penalties: High=100000, Medium=80000, Low=60000.
                priorityPenalty = [
                100000, 100000, 100000, 100000, 80000, 80000, 100000, 80000, 100000, 80000, 60000,
                100000, 100000, 100000, 100000, 80000, 80000, 100000, 80000, 100000, 80000, 60000,
                100000, 100000, 100000, 100000, 80000, 80000, 100000, 80000, 100000, 80000, 60000
                ];

                isWeekendShift = [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ];

                isLateShift = [
                0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0,
                0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0,
                0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0
                ];

                isNightShift = [
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
                ];

                // roleEligible[e][s]: role, transferable skill, night permission, and weekend permission only.
                roleEligible = [
                [1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,1,1,0,0,0,0,0,0, 0,0,0,1,1,0,0,0,0,0,0, 0,0,0,1,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,1,0,0,0, 0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,1,0,0, 0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1],
                [1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0]
                ];

                // windowOK[e][s]: shift fits the employee's stated window, except E011 D3 FO_EARLY is an approved cover exception.
                windowOK = [
                [1,0,0,1,1,0,0,0,0,0,1, 1,0,0,1,1,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,1,0,0,1,0, 0,1,0,0,0,0,1,0,0,1,0, 0,1,0,0,0,0,1,0,0,1,0],
                [0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,1,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,1,0,0,0,0,0,0,0],
                [1,0,0,1,1,0,0,0,0,0,1, 1,0,0,1,1,0,0,0,0,0,1, 1,0,0,1,1,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,1,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0],
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0],
                [0,1,0,0,0,0,1,0,0,1,0, 0,0,0,0,0,0,0,0,0,0,0, 0,1,0,0,0,0,1,0,0,1,0],
                [0,0,0,1,1,0,0,0,0,0,1, 0,0,0,1,1,0,0,0,0,0,1, 0,0,0,1,1,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0, 0,0,0,0,0,1,0,0,0,0,0]
                ];

                // hardUnavailable[e][s]: approved hard absences are forbidden. E011 D3 is soft, so it is not hard-unavailable.
                hardUnavailable = [
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0]
                ];

                // Preference match by preferred/requested shift type, with E011 D3 FO_EARLY treated as an approved match.
                prefMatch = [
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0],
                [0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0],
                [0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0, 0,0,1,0,0,0,0,0,0,0,0],
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0],
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0],
                [0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0],
                [0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0],
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0],
                [0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0, 0,1,0,0,0,0,1,1,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,0,0,1],
                [1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0, 1,0,0,1,1,1,0,0,1,0,0]
                ];

                // Requested day-off indicator; hard and soft requests are separated by hardUnavailable above.
                requestedOff = [
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1],
                [0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0]
                ];

                prefMismatchPenalty = 20;
                underTargetPenalty = 2;
                overTargetPenalty = 8;
                lateFairnessPenalty = 3;
                nightFairnessPenalty = 5;
                weekendFairnessPenalty = 4;
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_portfolio_diversification(self):
        """
        Test the Restless Multi-Armed Bandit (RMAB) LP relaxation with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                // -----------------------------------------------------------------------------
                // Multi-stage scenario-based stochastic portfolio diversification
                // -----------------------------------------------------------------------------
                // A portfolio is allocated across assets over a scenario tree.
                // Node-based decisions represent rebalancing decisions made after uncertainty
                // is revealed up to that node, which naturally enforces nonanticipativity.
                // The objective is to maximize expected terminal wealth at leaf nodes while
                // limiting concentration in any single asset at nonterminal nodes.
                // -----------------------------------------------------------------------------

                // -----------------------------
                // Sets and tuple definitions
                // -----------------------------

                {string} Assets = ...;              // Asset identifiers
                int nbNodes = ...;                  // Number of nodes in the scenario tree
                range Nodes = 1..nbNodes;           // Node index set

                tuple Arc {                         // Directed arc from parent node to child node
                int parent;
                int child;
                };

                {Arc} Arcs = ...;                   // Set of arcs in the scenario tree

                // -----------------------------
                // Parameters
                // -----------------------------

                int root = ...;                     // Root node index
                boolean isLeaf[Nodes] = ...;        // True if node n is a terminal node
                float prob[Nodes] = ...;            // Probability of node n if terminal; 0 for nonterminal nodes
                float ret[Arcs][Assets] = ...;      // Gross return factor for asset a along arc ar
                float initHold[Assets] = ...;       // Initial holdings before any trading at the root
                float transCost = ...;              // Proportional transaction cost rate
                float maxShare = ...;               // Maximum allowed asset share at nonterminal nodes

                // -----------------------------
                // Decision variables
                // -----------------------------

                dvar float+ h[Nodes][Assets];       // Post-trade holdings at node n in asset a
                dvar float+ buy[Nodes][Assets];     // Amount of asset a purchased at node n
                dvar float+ sell[Nodes][Assets];    // Amount of asset a sold at node n
                dvar float+ wealth[Nodes];          // Wealth at terminal nodes; forced to 0 at nonterminal nodes

                // -----------------------------
                // Objective
                // -----------------------------

                // Maximize expected terminal wealth over all leaf nodes.
                maximize ExpectedTerminalWealth:
                sum(n in Nodes : isLeaf[n]) prob[n] * wealth[n];

                // -----------------------------
                // Constraints
                // -----------------------------
                subject to {

                // Root portfolio balance after initial rebalancing.
                forall(a in Assets)
                    RootHoldBalance: h[root][a] == initHold[a] + buy[root][a] - sell[root][a];

                // Root self-financing with proportional transaction costs.
                RootSelfFinancing:
                    sum(a in Assets) ((1 + transCost) * buy[root][a])
                    ==
                    sum(a in Assets) ((1 - transCost) * sell[root][a]);

                // Root sales cannot exceed initial holdings.
                forall(a in Assets)
                    RootSellLimit: sell[root][a] <= initHold[a];

                // Child-node holdings equal returned parent holdings plus local trades.
                forall(ar in Arcs, a in Assets)
                    NodeHoldBalance: h[ar.child][a] == ret[ar][a] * h[ar.parent][a] + buy[ar.child][a] - sell[ar.child][a];

                // Self-financing at each non-root node.
                forall(ar in Arcs)
                    NodeSelfFinancing:
                    sum(a in Assets) ((1 + transCost) * buy[ar.child][a])
                    ==
                    sum(a in Assets) ((1 - transCost) * sell[ar.child][a]);

                // Sales at a child node cannot exceed pre-trade available holdings.
                forall(ar in Arcs, a in Assets)
                    NodeSellLimit: sell[ar.child][a] <= ret[ar][a] * h[ar.parent][a];

                // Diversification cap at each nonterminal node.
                forall(n in Nodes, a in Assets : !isLeaf[n])
                    Diversification: h[n][a] <= maxShare * sum(b in Assets) h[n][b];

                // Terminal wealth at each leaf equals the value of the portfolio after the final returns.
                forall(ar in Arcs : isLeaf[ar.child])
                    TerminalWealthDef: wealth[ar.child] == sum(a in Assets) ret[ar][a] * h[ar.parent][a];

                // Wealth variable is zero at nonterminal nodes.
                forall(n in Nodes : !isLeaf[n])
                    NonLeafWealthZero: wealth[n] == 0;
                }
                """
            data_code = """
                Assets = { "StockA", "StockB", "BondC" };
                nbNodes = 7;
                Arcs = { <1,2>, <1,3>, <2,4>, <2,5>, <3,6>, <3,7> };
                root = 1;
                isLeaf = [ false, false, false, true, true, true, true ];
                prob = [ 0, 0, 0, 0.25, 0.25, 0.25, 0.25 ];
                initHold = [
                "StockA" 40,
                "StockB" 35,
                "BondC" 25
                ];
                transCost = 0.01;
                maxShare = 0.60;
                ret = [
                <1,2> [1.08, 1.03, 1.01],
                <1,3> [0.95, 1.06, 1.02],
                <2,4> [1.10, 1.02, 1.01],
                <2,5> [0.92, 1.04, 1.01],
                <3,6> [1.07, 0.98, 1.01],
                <3,7> [0.90, 1.08, 1.02]
                ];
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_RMAB_relaxation_tuples(self):
        """
        Test the Restless Multi-Armed Bandit (RMAB) LP relaxation with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                /*
                Restless Multi-Armed Bandit (RMAB) — small example via occupation-measure LP

                Intent (stationary average-cost LP relaxation with an activation budget):
                - Arms k evolve as controlled Markov chains with state i ∈ States and action a ∈ Actions.
                - Occupation measures x[k][i][a] represent the steady-state fraction of time arm k is in state i taking action a.
                - Per-arm constraints enforce a valid stationary distribution (normalization + flow balance).
                - A coupling constraint limits the expected number of Active arms per step (time-average budget).
                */

                /********************
                * Sets and indices *
                ********************/
                {string} Arms = ...;        // Arm identifiers (e.g., {"A1","A2"})
                {int}    States = ...;      // State labels (here assumed {1,2,...,S})
                {string} Actions = ...;     // Action labels (includes "Active")

                /****************
                * Input data   *
                ****************/
                param float cost[Arms][States][Actions] = ...;        // Immediate cost c(k,i,a)

                // Transition probabilities P(k,i,a,j): probability of next-state j given (k,i,a).
                param float P[Arms][States][Actions][States] = ...;

                param float Budget = ...;                             // Expected #Active arms per step

                /************************
                * Decision variables   *
                ************************/
                // x[k][i][a] ≥ 0: steady-state joint probability of (state=i, action=a) for arm k.
                dvar float+ x[Arms][States][Actions];

                /****************
                * Objective    *
                ****************/
                minimize AverageCost:
                // (OBJ) Long-run expected average cost across all arms
                sum(k in Arms, i in States, a in Actions) cost[k][i][a] * x[k][i][a];

                /****************
                * Constraints  *
                ****************/
                subject to {
                // (C1) ArmNormalize: each arm's occupation measures sum to 1.
                forall(k in Arms)
                    ArmNormalize:
                    sum(i in States, a in Actions) x[k][i][a] == 1;

                // (C2) ArmFlowBalance: steady-state flow balance for each arm and state.
                // Mass in state j equals mass transitioning into j.
                forall(k in Arms, j in States)
                    ArmFlowBalance:
                    sum(a in Actions) x[k][j][a]
                        ==
                    sum(i in States, a in Actions) x[k][i][a] * P[k][i][a][j];

                // (C3) BudgetActive: expected number of active arms per step is limited.
                BudgetActive:
                    sum(k in Arms, i in States) x[k][i]["Active"] <= Budget;

                // (C4) RowStochastic: each transition row is stochastic (sums to 1 over next-state).
                forall(k in Arms, i in States, a in Actions)
                    RowStochastic:
                    sum(j in States) P[k][i][a][j] == 1;
                }
                """
            data_code = """
                /*
                Small RMAB instance

                - 2 arms: A1, A2
                - 2 states: 1 (Good), 2 (Bad)
                - 2 actions: Passive, Active
                - Budget = 1 means: on average, at most one arm is Active each step.

                Data format note:
                - cost is keyed by <arm,state,action> tuples.
                - P is now a fully indexed 4D parameter keyed by <arm,state,action,next_state> tuples.
                */

                Arms = { "A1", "A2" };
                States = { 1, 2 };
                Actions = { "Passive", "Active" };

                Budget = 1;

                // Immediate cost c(k,i,a)
                cost = [
                <"A1",1,"Passive"> 0.0,
                <"A1",1,"Active">  0.5,
                <"A1",2,"Passive"> 3.0,
                <"A1",2,"Active">  3.5,

                <"A2",1,"Passive"> 0.0,
                <"A2",1,"Active">  0.5,
                <"A2",2,"Passive"> 4.0,
                <"A2",2,"Active">  4.5
                ];

                // Transition probabilities P(k,i,a,j)
                P = [
                <"A1",1,"Passive",1> 0.80,
                <"A1",1,"Passive",2> 0.20,
                <"A1",1,"Active",1>  0.95,
                <"A1",1,"Active",2>  0.05,
                <"A1",2,"Passive",1> 0.20,
                <"A1",2,"Passive",2> 0.80,
                <"A1",2,"Active",1>  0.60,
                <"A1",2,"Active",2>  0.40,

                <"A2",1,"Passive",1> 0.70,
                <"A2",1,"Passive",2> 0.30,
                <"A2",1,"Active",1>  0.90,
                <"A2",1,"Active",2>  0.10,
                <"A2",2,"Passive",1> 0.10,
                <"A2",2,"Passive",2> 0.90,
                <"A2",2,"Active",1>  0.50,
                <"A2",2,"Active",2>  0.50
                ];
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_RMAB_relaxation_dense(self):
        """
        Test the Restless Multi-Armed Bandit (RMAB) LP relaxation with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                /*
                Restless Multi-Armed Bandit (RMAB) — small example via occupation-measure LP

                Intent (stationary average-cost LP relaxation with an activation budget):
                - Arms k evolve as controlled Markov chains with state i ∈ States and action a ∈ Actions.
                - Occupation measures x[k][i][a] represent the steady-state fraction of time arm k is in state i taking action a.
                - Per-arm constraints enforce a valid stationary distribution (normalization + flow balance).
                - A coupling constraint limits the expected number of Active arms per step (time-average budget).
                */

                /********************
                * Sets and indices *
                ********************/
                {string} Arms = ...;        // Arm identifiers (e.g., {"A1","A2"})
                {int}    States = ...;      // State labels (here assumed {1,2,...,S})
                {string} Actions = ...;     // Action labels (includes "Active")

                /****************
                * Input data   *
                ****************/
                param float cost[Arms][States][Actions] = ...;        // Immediate cost c(k,i,a)

                // Transition probabilities P(k,i,a,j): probability of next-state j given (k,i,a).
                param float P[Arms][States][Actions][States] = ...;

                param float Budget = ...;                             // Expected #Active arms per step

                /************************
                * Decision variables   *
                ************************/
                // x[k][i][a] ≥ 0: steady-state joint probability of (state=i, action=a) for arm k.
                dvar float+ x[Arms][States][Actions];

                /****************
                * Objective    *
                ****************/
                minimize AverageCost:
                // (OBJ) Long-run expected average cost across all arms
                sum(k in Arms, i in States, a in Actions) cost[k][i][a] * x[k][i][a];

                /****************
                * Constraints  *
                ****************/
                subject to {
                // (C1) ArmNormalize: each arm's occupation measures sum to 1.
                forall(k in Arms)
                    ArmNormalize:
                    sum(i in States, a in Actions) x[k][i][a] == 1;

                // (C2) ArmFlowBalance: steady-state flow balance for each arm and state.
                // Mass in state j equals mass transitioning into j.
                forall(k in Arms, j in States)
                    ArmFlowBalance:
                    sum(a in Actions) x[k][j][a]
                        ==
                    sum(i in States, a in Actions) x[k][i][a] * P[k][i][a][j];

                // (C3) BudgetActive: expected number of active arms per step is limited.
                BudgetActive:
                    sum(k in Arms, i in States) x[k][i]["Active"] <= Budget;

                // (C4) RowStochastic: each transition row is stochastic (sums to 1 over next-state).
                forall(k in Arms, i in States, a in Actions)
                    RowStochastic:
                    sum(j in States) P[k][i][a][j] == 1;
                }
                """
            data_code = """
                /*
                Small RMAB instance (dense array form to match declarations)

                Index orders must match the model declarations:
                - cost[Arms][States][Actions]
                - P[Arms][States][Actions][States]

                Given:
                Arms = {"A1","A2"} (assume this order)
                States = {1,2}
                Actions = {"Passive","Active"}
                */

                Arms = { "A1", "A2" };
                States = { 1, 2 };
                Actions = { "Passive", "Active" };

                Budget = 1;

                // cost[k][i][a]
                cost = [
                [ // A1
                    [0.0, 0.5],  // state 1: Passive, Active
                    [3.0, 3.5]   // state 2: Passive, Active
                ],
                [ // A2
                    [0.0, 0.5],  // state 1
                    [4.0, 4.5]   // state 2
                ]
                ];

                // P[k][i][a][j]
                P = [
                [ // A1
                    [ // state 1
                    [0.80, 0.20], // Passive -> (j=1,j=2)
                    [0.95, 0.05]  // Active  -> (j=1,j=2)
                    ],
                    [ // state 2
                    [0.20, 0.80], // Passive
                    [0.60, 0.40]  // Active
                    ]
                ],
                [ // A2
                    [ // state 1
                    [0.70, 0.30], // Passive
                    [0.90, 0.10]  // Active
                    ],
                    [ // state 2
                    [0.10, 0.90], // Passive
                    [0.50, 0.50]  // Active
                    ]
                ]
                ];
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_column_generation(self):
        """
        Test a column generation model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                /*
                Column Generation in Simple MILP Terms (Cutting-Stock / Pattern Selection)

                Key idea
                - Master (RMP): choose how many stock rolls to cut with each currently-known pattern p.
                - A “column” = a pattern p (its vector of item counts a[p][i]).
                - Pricing (knapsack): using dual prices from the RMP-LP, search for a new pattern with
                negative reduced cost.

                Modeling note
                - Column generation is an algorithm (iterate RMP-LP -> duals -> pricing -> add column).
                A single static MILP cannot “add columns”. Here we keep BOTH models in one file for
                explanation, but we activate one side at a time using a ground switch RunPricing.
                - If RunPricing = 0: solve the master MILP.
                - If RunPricing = 1: solve the pricing knapsack (with mock duals supplied in data).

                PyOPL note
                - We gate constraints with a boolean OR pattern:
                    (RunPricing != 0) || ( constraint )
                which enforces the constraint only when RunPricing == 0 (and similarly for pricing).
                */

                // ----------------------------
                // Sets and indices
                // ----------------------------

                {string} I = ...;          // item types
                {string} P = ...;          // currently available patterns (columns)

                // ----------------------------
                // Parameters (data inputs)
                // ----------------------------

                param int L = ...;                 // stock roll length (knapsack capacity)
                param int itemLen[I] = ...;        // length per item type (renamed from reserved "len")
                param int demand[I] = ...;         // required quantity of each item

                // Pattern definition: a[p][i] = number of items i produced by one roll using pattern p
                param int a[P][I] = ...;

                // Dual prices from the *LP relaxation* of the master demand constraints (provided as data here)
                param float dual[I] = ...;

                // Ground switch: 0 = solve master (RMP), 1 = solve pricing (knapsack)
                param int RunPricing = ...;

                // ----------------------------
                // Decision variables
                // ----------------------------

                // [Var_x] Master decision: x[p] = number of rolls cut with pattern p (integer number of rolls)
                dvar int+ x[P];

                // [Var_y] Pricing decision: y[i] = counts of items i in a candidate new cutting pattern (integer)
                dvar int+ y[I];

                // ----------------------------
                // Derived expressions (pricing)
                // ----------------------------

                // [Dexpr_PricingValue] Dual value of a candidate pattern (what the master would “pay” for this column)
                dexpr float PricingValue = sum(i in I) dual[i] * y[i];

                // [Dexpr_ReducedCost] Reduced cost for a master minimization with cost 1 per roll:
                // rc = 1 - sum_i dual[i]*y[i]. If rc < 0 (equivalently PricingValue > 1), add the column.
                dexpr float ReducedCost = 1 - PricingValue;

                // ----------------------------
                // Objective (robust gating; avoids ternary objective issues)
                // ----------------------------

                // [Obj] If solving master, minimize rolls used; if solving pricing, minimize reduced cost.
                // Since RunPricing is a ground parameter, (RunPricing==1) and (RunPricing!=1) are ground 0/1.
                minimize Obj:
                (RunPricing == 1) * ReducedCost + (RunPricing != 1) * (sum(p in P) x[p]);

                subject to {
                // ----------------------------
                // Master (RMP) constraints (active only when RunPricing == 0)
                // ----------------------------

                // [Master_DemandCover] Meet each item demand using the current pattern set P
                forall(i in I)
                    Master_DemandCover: (RunPricing != 0) || (sum(p in P) a[p][i] * x[p] >= demand[i]);

                // [Master_PatternFeasible] Data consistency: each listed pattern must fit within one roll
                forall(p in P)
                    Master_PatternFeasible: (RunPricing != 0) || (sum(i in I) itemLen[i] * a[p][i] <= L);

                // ----------------------------
                // Pricing (knapsack) constraints (active only when RunPricing == 1)
                // ----------------------------

                // [Pricing_Capacity] Candidate new pattern must fit within one roll
                Pricing_Capacity: (RunPricing != 1) || (sum(i in I) itemLen[i] * y[i] <= L);

                // [Pricing_NegativeReducedCostTest] Look for an improving column: PricingValue > 1
                // Encode strictness via a small epsilon.
                Pricing_NegativeReducedCostTest: (RunPricing != 1) || (PricingValue >= 1.000001);
                }
                """
            data_code = """
                // Small mock instance: 3 item types, roll length 10.
                // Items: A (len 2), B (len 3), C (len 5)
                // Demand: need 4 A, 3 B, 2 C

                I = { "A", "B", "C" };
                L = 10;

                itemLen = [
                "A" 2,
                "B" 3,
                "C" 5
                ];

                demand = [
                "A" 4,
                "B" 3,
                "C" 2
                ];

                // Start with a small restricted set of patterns (columns).
                // p1: 5A (2*5=10)
                // p2: 3B (3*3=9)
                // p3: 2C (5*2=10)
                // p4: 2A + 2B (2*2 + 3*2 = 10)
                P = { "p1", "p2", "p3", "p4" };

                a = [
                "p1" [ 5, 0, 0 ],
                "p2" [ 0, 3, 0 ],
                "p3" [ 0, 0, 2 ],
                "p4" [ 2, 2, 0 ]
                ];

                // Mock dual prices (as if from solving the RMP LP relaxation).
                // Pricing tries to pack high-dual items into one roll.
                dual = [
                "A" 0.10,
                "B" 0.35,
                "C" 0.55
                ];

                // Choose which submodel to run:
                // 0 = solve master MILP (pattern selection)
                // 1 = solve pricing knapsack (find an improving column)
                RunPricing = 1;
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_TOPSIS(self):
        """
        Test the TOPSIS problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                // -----------------------------
                // Dimensions
                // -----------------------------
                int NA = ...;                      // number of alternatives
                int NC = ...;                      // number of criteria

                range Alternatives = 1..NA;
                range Criteria     = 1..NC;

                // Optional labels (for reporting)
                param string AltName[Alternatives] = ...; // e.g., ["Phone A","Phone B","Phone C"]
                param string CritName[Criteria]    = ...; // e.g., ["Price","Camera"]

                // -----------------------------
                // Inputs
                // -----------------------------
                param float   X[Alternatives][Criteria] = ...;   // decision matrix
                param float   w[Criteria] = ...;                 // weights (nonnegative, typically sum to 1)
                param boolean is_benefit[Criteria] = ...;        // true = benefit, false = cost

                // -----------------------------
                // TOPSIS computations (all ground)
                // -----------------------------
                // Step 2: Vector normalization denominators per criterion
                param float denom[c in Criteria] = sqrt( sum(i in Alternatives) ( X[i][c] * X[i][c] ) );

                // Normalized matrix
                param float r[i in Alternatives][c in Criteria] = X[i][c] / denom[c];

                // Step 3: Weighted normalized matrix
                param float v[i in Alternatives][c in Criteria] = r[i][c] * w[c];

                // Step 4: Per-criterion extrema of v
                param float v_max[c in Criteria] = max( i in Alternatives ) ( v[i][c] );
                param float v_min[c in Criteria] = min( i in Alternatives ) ( v[i][c] );

                // Positive/Negative Ideal Solutions per criterion
                param float v_plus[c in Criteria]  = (is_benefit[c]) ? v_max[c] : v_min[c];
                param float v_minus[c in Criteria] = (is_benefit[c]) ? v_min[c] : v_max[c];

                // Step 5: Distances (squared, then Euclidean) from PIS and NIS
                param float Splus[i in Alternatives]  = sum(c in Criteria) ( (v[i][c] - v_plus[c])  * (v[i][c] - v_plus[c]) );
                param float Sminus[i in Alternatives] = sum(c in Criteria) ( (v[i][c] - v_minus[c]) * (v[i][c] - v_minus[c]) );

                param float dplus[i in Alternatives]  = sqrt( Splus[i] );
                param float dminus[i in Alternatives] = sqrt( Sminus[i] );

                // Step 6: TOPSIS performance score Ci in [0,1]
                param float Ci[i in Alternatives] = dminus[i] / ( dplus[i] + dminus[i] );

                // -----------------------------
                // Decision variables
                // -----------------------------
                // y[i] = 1 if alternative i is selected
                dvar boolean y[Alternatives];

                // -----------------------------
                // Objective: pick the alternative with the highest TOPSIS score
                // -----------------------------
                maximize choose_best: sum(i in Alternatives) Ci[i] * y[i];

                // -----------------------------
                // Constraints
                // -----------------------------
                subject to {
                // Pick exactly one alternative
                pick_one: sum(i in Alternatives) y[i] == 1;
                }
                """
            data_code = """
                NA = 3;
                NC = 2;

                AltName = ["Phone A", "Phone B", "Phone C"];
                CritName = ["Price", "Camera"];

                // Decision matrix X: rows are alternatives, columns are criteria
                // Price is a cost (lower is better); Camera is a benefit (higher is better)
                X = [ [800, 7],    // Phone A
                    [600, 4],    // Phone B
                    [1200, 10] ];// Phone C

                // Weights: more importance on Camera
                w = [0.4, 0.6];

                // Orientation per criterion: [Price is cost -> false, Camera is benefit -> true]
                is_benefit = [false, true];
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    @unittest.skip("this test is cumbersome to run")
    def test_asset_location(self):
        """
        Test the vehicle routing problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        # Set scipy codegen logger to INFO only for this test
        import logging

        _scipy_logger = logging.getLogger("pyopl.scipy_codegen_csc")
        _prev_level = _scipy_logger.level
        _scipy_logger.setLevel(logging.INFO)
        try:
            model_code = """
                // Cell-based 8-neighbor MILP for A--B routing with per-asset zones (PyOPL)

                // Tuple types for grid constructs
                // Cell: grid coordinate (row i, col j)
                // Arc:  directed neighbor arc u=(ui,uj) -> v=(vi,vj)
                // Edge: undirected neighbor edge endpoints (canonicalized order)
                tuple Cell { int i; int j; }
                tuple Arc  { int ui; int uj; int vi; int vj; }
                tuple Edge { int i1; int j1; int i2; int j2; }

                // Grid and type dimensions
                param int NR = ...;                               // number of rows
                param int NC = ...;                               // number of cols
                param int NT = ...;                               // number of asset types

                range I = 1..NR;                                  // row indices
                range J = 1..NC;                                  // col indices
                range Types = 1..NT;                              // asset types {1..NT}

                // Cells, directed arcs (8-neighbor), and undirected edges (canonicalized)
                {Cell} V = { <i, j> | i in I, j in J };

                {Arc} A = {
                <i, j, ip, jp>
                | i in I, j in J, ip in I, jp in J
                : ((i != ip) || (j != jp))
                    && ((ip == i - 1) || (ip == i) || (ip == i + 1))
                    && ((jp == j - 1) || (jp == j) || (jp == j + 1))
                };

                {Edge} E = {
                <i, j, ip, jp>
                | i in I, j in J, ip in I, jp in J
                : ((i != ip) || (j != jp))
                    && ((ip == i - 1) || (ip == i) || (ip == i + 1))
                    && ((jp == j - 1) || (jp == j) || (jp == j + 1))
                    && ((i < ip) || ((i == ip) && (j < jp)))
                };

                // Terminals as scalar coordinates (avoid tuple in .dat)
                param int Ai = ...;  // row of terminal A
                param int Aj = ...;  // col of terminal A
                param int Bi = ...;  // row of terminal B
                param int Bj = ...;  // col of terminal B

                // Geometry weights and cost parameters
                param float lambda_w = ...;                        // weight on geometric length
                param float eps = ...;                             // small penalty on selected cells

                // Per-type base build costs; per-cell cost derived below
                param float base[Types] = ...;                     // base cost per type

                // Arc geometric weights: 1 for cardinal, sqrt(2) for diagonal
                param float w[a in A] = (((a.ui != a.vi) && (a.uj != a.vj)) ? sqrt(2) : 1);

                // Connectivity supply (+1 at A), demand (-1 at B), zero elsewhere
                param float b[v in V] = (((v.i == Ai && v.j == Aj) ? 1 : ((v.i == Bi && v.j == Bj) ? -1 : 0)));

                // Zone definitions to compute allowed[v,t]
                param int Z1_i_lo = ...;  param int Z1_i_hi = ...;   // mandatory type-1 rectangle rows
                param int Z1_j_lo = ...;  param int Z1_j_hi = ...;   // mandatory type-1 rectangle cols
                param int Z2_i_lo = ...;  param int Z2_i_hi = ...;   // mandatory type-2 rectangle rows
                param int Z2_j_lo = ...;  param int Z2_j_hi = ...;   // mandatory type-2 rectangle cols
                param int RS_lo = ...;     param int RS_hi = ...;     // row stripe forbidding type-6

                // Per-cell, per-type allowance (boolean) built from simple zones
                // - Zone1: only type 1 allowed
                // - Zone2: only type 2 allowed
                // - Diagonal band (i==j or adjacent): type 3 forbidden
                // - Row stripe [RS_lo..RS_hi]: type 6 forbidden
                param boolean allowed[v in V][t in Types] =
                (((v.i >= Z1_i_lo) && (v.i <= Z1_i_hi) && (v.j >= Z1_j_lo) && (v.j <= Z1_j_hi))
                    ? (t == 1)
                    : (((v.i >= Z2_i_lo) && (v.i <= Z2_i_hi) && (v.j >= Z2_j_lo) && (v.j <= Z2_j_hi))
                        ? (t == 2)
                        : (((v.j == v.i) || (v.j == v.i + 1) || (v.i == v.j + 1))
                            ? (t != 3)
                            : ((((v.i >= RS_lo) && (v.i <= RS_hi)) ? (t != 6) : (true))))));

                // Per-cell, per-type build cost (computed); avoids external data for c
                // cost[v,t] = base[t] + small location term
                dexpr float cost[v in V][t in Types] = base[t] + 0.01 * (v.i + v.j);

                // Decision variables
                // x[v] = 1 if cell v is on the route
                // y[v,t] = 1 if asset t is used in cell v
                // f[a] >= 0 is unit flow on arc a (for A->B connectivity)
                // z[e] = 1 if undirected edge e is used by the chain (optional simple-path enforcement)
                dvar boolean x[V];
                dvar boolean y[V][Types];
                dvar float+  f[A];
                dvar boolean z[E];

                // Objective: build cost + geometric length + small node penalty
                minimize obj:
                    sum(v in V, t in Types) cost[v][t] * y[v][t]
                + lambda_w * sum(a in A) w[a] * f[a]
                + eps * sum(v in V) x[v]
                ;

                subject to {
                // C1: include terminals (force A and B cells selected)
                C1A: sum(v in V: v.i == Ai && v.j == Aj) x[v] == 1;
                C1B: sum(v in V: v.i == Bi && v.j == Bj) x[v] == 1;

                // C2: exactly one asset iff the cell is selected
                forall(v in V) C2_assign: sum(t in Types) y[v][t] == x[v];

                // C3: respect allowed asset types per cell
                forall(v in V, t in Types) C3_allowed: y[v][t] <= allowed[v][t];

                // C4: flow conservation enforces A->B connectivity
                forall(v in V)
                    C4_flow:
                    (sum(a in A: a.ui == v.i && a.uj == v.j) f[a])
                    - (sum(a in A: a.vi == v.i && a.vj == v.j) f[a])
                    == b[v];

                // C5: flow can leave node u only if u is selected
                forall(a in A)
                    C5_cap_u: f[a] <= sum(v in V: v.i == a.ui && v.j == a.uj) x[v];

                // C6: flow can enter node v only if v is selected
                forall(a in A)
                    C6_cap_v: f[a] <= sum(v in V: v.i == a.vi && v.j == a.vj) x[v];

                // Optional simple-path (non-branching) chain constraints
                // S1: any flow across {u,v} activates that undirected edge
                forall(e in E, a in A: a.ui == e.i1 && a.uj == e.j1 && a.vi == e.i2 && a.vj == e.j2)
                    S1_dir1: f[a] <= z[e];
                forall(e in E, a in A: a.ui == e.i2 && a.uj == e.j2 && a.vi == e.i1 && a.vj == e.j1)
                    S1_dir2: f[a] <= z[e];

                // S2: used edge implies both endpoint cells are selected
                forall(e in E) S2_u: z[e] <= sum(v in V: v.i == e.i1 && v.j == e.j1) x[v];
                forall(e in E) S2_v: z[e] <= sum(v in V: v.i == e.i2 && v.j == e.j2) x[v];

                // S3: degree-1 at terminals
                S3_A: sum(e in E: ((e.i1 == Ai && e.j1 == Aj) || (e.i2 == Ai && e.j2 == Aj))) z[e] == 1;
                S3_B: sum(e in E: ((e.i1 == Bi && e.j1 == Bj) || (e.i2 == Bi && e.j2 == Bj))) z[e] == 1;

                // S4: degree-2 at internal selected cells
                forall(v in V: !((v.i == Ai && v.j == Aj) || (v.i == Bi && v.j == Bj)))
                    S4_deg2:
                    sum(e in E: ((e.i1 == v.i && e.j1 == v.j) || (e.i2 == v.i && e.j2 == v.j))) z[e] == 2 * x[v];
                }
                """
            data_code = """
                NR = 4;
                NC = 4;
                NT = 2;
                Ai = 2;
                Aj = 2;
                Bi = 3;
                Bj = 3;
                lambda_w = 0.5;
                eps = 0.001;
                base = [1.0, 1.2];
                Z1_i_lo = 1;  Z1_i_hi = 2;
                Z1_j_lo = 1;  Z1_j_hi = 13;
                Z2_i_lo = 1;  Z2_i_hi = 3;
                Z2_j_lo = 2; Z2_j_hi = 4;
                RS_lo = 1;   RS_hi = 2;
                """
            import os
            import tempfile

            from pyopl.pyopl_core import solve

            results = {}
            for solver in ("scipy", "gurobi"):
                with (
                    tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                    tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
                ):
                    tmp_mod.write(model_code)
                    tmp_mod.flush()
                    tmp_dat.write(data_code)
                    tmp_dat.flush()
                    model_file = tmp_mod.name
                    data_file = tmp_dat.name
                try:
                    result = solve(model_file, data_file, solver=solver)
                    self.assertNotEqual(result["status"], "FAILED")
                    results[solver] = result
                finally:
                    os.remove(model_file)
                    os.remove(data_file)

            # If both solvers are infeasible, test passes
            if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
                return  # Test passes

            # Otherwise, require both to be optimal and compare objectives
            self.assertEqual(results["scipy"]["status"], "OPTIMAL")
            self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
            self.assertIn("objective_value", results["scipy"])
            self.assertIn("objective_value", results["gurobi"])
            self.assertAlmostEqual(
                results["scipy"]["objective_value"],
                results["gurobi"]["objective_value"],
                places=6,
            )
        finally:
            # Restore previous level
            _scipy_logger.setLevel(_prev_level)

    def test_vrp(self):
        """
        Test the vehicle routing problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            /*
            PyOPL Model: Vehicle Routing Problem (VRP) with MTZ Subtour Elimination
            - Cities are nodes with demand and possible depot(s)
            - Vehicles have capacity
            - Arcs model feasible routes (with distances)
            - Decision variables:
                x[a]: binary, whether arc a in ARCS is used
                u[n]: continuous, load on vehicle upon arrival at node n (for subtour elimination)
            - Objective: Minimize total travel distance
            - Subtour elimination: Miller-Tucker-Zemlin (MTZ) constraints
            */

            tuple Node {
            string name;
            float demand;
            boolean is_depot;
            };

            tuple Arc {
            string from;
            string to;
            float distance;
            };

            {Node} NODES;
            {Arc}  ARCS;
            {string} NODE_NAMES;
            {string} DEPOT_NAMES;

            param int NUM_VEHICLES;
            param float VEHICLE_CAPACITY;
            param float demand[NODE_NAMES];
            param boolean is_depot[NODE_NAMES];
            param float distance[ARCS];

            // Decision variables
            // x[a]: 1 if arc a is used, 0 otherwise
            // u[n]: load assigned to node n (for MTZ subtour elimination), only for non-depot nodes

            dvar boolean x[ARCS];
            dvar float u[NODE_NAMES];

            minimize total_distance: sum(a in ARCS) distance[a] * x[a];

            subject to {
            // 1. Each customer node is entered exactly once (except depots)
            forall(n in NODE_NAMES : !is_depot[n])
                sum(a in ARCS : a.to == n) x[a] == 1;

            // 2. Each customer node is exited exactly once (except depots)
            forall(n in NODE_NAMES : !is_depot[n])
                sum(a in ARCS : a.from == n) x[a] == 1;

            // 3. Vehicle count constraint at depots: Outflow == NUM_VEHICLES, Inflow == NUM_VEHICLES
            forall(d in DEPOT_NAMES)
                sum(a in ARCS : a.from == d) x[a] == NUM_VEHICLES;
            forall(d in DEPOT_NAMES)
                sum(a in ARCS : a.to == d) x[a] == NUM_VEHICLES;

            // 4. Subtour elimination (MTZ) - only for non-depot nodes
            // For all i != j, i and j are not depots
            forall(i in NODE_NAMES : !is_depot[i])
                u[i] >= demand[i];
            forall(i in NODE_NAMES : !is_depot[i])
                u[i] <= VEHICLE_CAPACITY;

            forall(i in NODE_NAMES : !is_depot[i])
                forall(j in NODE_NAMES : (!is_depot[i] && !is_depot[j] && i != j)) {
                // There may be multiple arcs between i and j (in ARCS). Apply MTZ to all arcs from i to j.
                forall(a in ARCS : a.from == i && a.to == j) {
                    u[i] - u[j] + VEHICLE_CAPACITY * x[a] <= VEHICLE_CAPACITY - demand[j];
                }
                }

            // 5. No self-loops
            forall(a in ARCS : a.from == a.to)
                x[a] == 0;
            }
            """
        data_code = """
            // Nodes: name, demand, is_depot
            NODES = {
            <"Depot", 0.0,  true>,
            <"A",     2.0, false>,
            <"B",     1.5, false>,
            <"C",     1.0, false>
            };

            // Arcs: fully connected directed network (except self-loops)
            ARCS = {
            <"Depot", "A",  4.0>,
            <"Depot", "B",  6.0>,
            <"Depot", "C",  8.0>,
            <"A",    "Depot", 4.0>,
            <"B",    "Depot", 6.0>,
            <"C",    "Depot", 8.0>,
            <"A",     "B",   5.0>,
            <"A",     "C",   7.0>,
            <"B",     "A",   5.0>,
            <"B",     "C",   4.0>,
            <"C",     "A",   7.0>,
            <"C",     "B",   4.0>
            };

            // Sets of node names and depot names
            NODE_NAMES = { "Depot", "A", "B", "C" };
            DEPOT_NAMES = { "Depot" };

            // Demand and is_depot per node name (mapping strings)
            demand = [
            "Depot" 0.0,
            "A"     2.0,
            "B"     1.5,
            "C"     1.0
            ];
            is_depot = [
            "Depot" true,
            "A"     false,
            "B"     false,
            "C"     false
            ];

            // Distance for each arc
            distance = [
            <"Depot","A",4.0> 4.0,
            <"Depot","B",6.0> 6.0,
            <"Depot","C",8.0> 8.0,
            <"A","Depot",4.0> 4.0,
            <"B","Depot",6.0> 6.0,
            <"C","Depot",8.0> 8.0,
            <"A","B",5.0>     5.0,
            <"A","C",7.0>     7.0,
            <"B","A",5.0>     5.0,
            <"B","C",4.0>     4.0,
            <"C","A",7.0>     7.0,
            <"C","B",4.0>     4.0
            ];

            // Number of vehicles and vehicle capacity
            NUM_VEHICLES = 2;
            VEHICLE_CAPACITY = 3.0;
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_vrp_2(self):
        """
        Test the vehicle routing problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            int n = ...;
            range Nodes = 0..n;
            int m = ...;
            int Q = ...;
            int dist[Nodes][Nodes];
            int demand[1..n];

            // Decision variables
            dvar boolean x[Nodes][Nodes];
            dvar int+ u[Nodes];

            // Objective: minimize total distance
            minimize
            sum(i in Nodes, j in Nodes) dist[i][j] * x[i][j];

            subject to {
            // No self loops
            forall(i in Nodes) x[i][i] == 0;

            // Each customer visited exactly once
            forall(i in 1..n)
                sum(j in Nodes) x[i][j] == 1;
            forall(j in 1..n)
                sum(i in Nodes) x[i][j] == 1;

            // Depot departures/arrivals equal to number of vehicles
            sum(j in 1..n) x[0][j] == m;
            sum(i in 1..n) x[i][0] == m;

            // MTZ subtour elimination (Miller-Tucker-Zemlin)
            forall(i in 1..n, j in 1..n: i != j)
                u[i] - u[j] + Q * x[i][j] <= Q - demand[j];

            // capacity bounds and depot load
            forall(i in 1..n) {
                demand[i] <= u[i];
                u[i] <= Q;
            }
            u[0] == 0;
            }
            """
        data_code = """
            n = 4;
            m = 2;
            Q = 15;

            // distance matrix indexed 0..n (depot = 0)
            dist = [
            [0, 4, 6, 9, 7],
            [4, 0, 5, 7, 3],
            [6, 5, 0, 4, 6],
            [9, 7, 4, 0, 5],
            [7, 3, 6, 5, 0]
            ];

            // demands for customers 1..n
            // customer indices: 1..4

            demand = [2, 4, 2, 5];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_stochastic_lot_sizing(self):
        """
        Test the stochastic lot-sizing problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Stochastic Lot-Sizing with Inventory and Backlog (Scenario-based, PyOPL)
            // Literate formulation with comments for clarity

            // -------------------------------
            // Sets and index ranges
            // -------------------------------
            {string} Scenarios = ...;                                       // Set of scenario labels
            param int T = ...;                                              // Number of periods
            range Periods = 1..T;                                           // Periods 1,2,...,T
            range ExtPeriods = 0..T;                                        // Extended periods including initial state (0)

            // -------------------------------
            // Parameters
            // -------------------------------
            param float p[Scenarios];                                       // Scenario probability (nonnegative, sum=1)
            param float demand[Scenarios][Periods];                         // Demand in each period for each scenario
            param float init_inventory;                                     // Initial inventory at time 0
            param float init_backlog;                                       // Initial backlog at time 0
            param float order_cost;                                         // Per-unit order cost
            param float holding_cost;                                       // Per-unit inventory holding cost per period
            param float backlog_cost;                                       // Per-unit backlog cost per period
            param float order_cap;                                          // Maximum order per period
            param float terminal_value;                                     // Terminal adjustment value per unit (salvage/penalty)

            // -------------------------------
            // Decision Variables
            // -------------------------------
            dvar float+ x[Scenarios][Periods];                              // Order quantity in period t under scenario s
            // State variables (indexed by periods 0..T, i.e. ExtPeriods)
            dvar float+ inventory[Scenarios][ExtPeriods];                   // Inventory at end of each period
            // 'inventory[s][t]' = ending inventory of scenario s after period t
            // inventory[s][0] = initial, known

            dvar float+ backlog[Scenarios][ExtPeriods];                     // Backlog at end of each period
            // 'backlog[s][t]' = ending backlog of scenario s after period t
            // backlog[s][0] = initial, known

            // -------------------------------
            // Objective: expected total cost
            // -------------------------------
            // (Ordering, holding, backlog, and terminal adjustment. Expected over scenarios.)
            minimize expected_total_cost:
            sum(s in Scenarios) p[s] * (
                sum(t in Periods) (
                order_cost * x[s][t]                        // variable ordering cost
                + holding_cost * inventory[s][t]            // end-of-period inventory holding cost
                + backlog_cost * backlog[s][t]              // end-of-period backlog cost
                )
                + terminal_value * (inventory[s][T] - backlog[s][T])    // salvage/penalty at time horizon end
            );

            subject to {
            // (C1) Initial state constraints: inventory and backlog at time 0 for all scenarios
            forall(s in Scenarios) {
                inventory[s][0] == init_inventory;
                backlog[s][0] == init_backlog;
            }

            // (C2) Inventory and backlog evolution equations
            forall(s in Scenarios, t in Periods) {
                // After orders and demand are realized in period t
                inventory[s][t] >= inventory[s][t-1] - backlog[s][t-1] + x[s][t] - demand[s][t];
                backlog[s][t] >= -(inventory[s][t-1] - backlog[s][t-1] + x[s][t] - demand[s][t]);
            }

            // (C3) Order capacity constraints
            forall(s in Scenarios, t in Periods)
                x[s][t] <= order_cap;

            // (C4) Nonanticipativity constraints
            // In period t: If two scenarios s1 and s2 have same demand history up to t-1,
            // then their order decisions in t must match (since decisions are made before t's demand).
            forall(t in Periods, s1 in Scenarios, s2 in Scenarios :
                (t == 1) || (sum(k in 1..(t-1)) (demand[s1][k] == demand[s2][k]) == t-1)
            ) x[s1][t] == x[s2][t];

            // (C5) Variable domains (float+) ensure nonnegativity; order, inventory, and backlog >= 0
            }
            """
        data_code = """
            T = 3;
            Scenarios = { "S1", "S2", "S3", "S4" };
            p = [ "S1" 0.25, "S2" 0.25, "S3" 0.25, "S4" 0.25 ];
            demand = [
            "S1" [80, 70, 60],
            "S2" [80, 70, 140],
            "S3" [80, 110, 60],
            "S4" [130, 110, 140]
            ];
            init_inventory = 15;
            init_backlog = 0;
            order_cost = 5.0;
            holding_cost = 1.0;
            backlog_cost = 9.0;
            order_cap = 120.0;
            terminal_value = 0.0;
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_newsvendor(self):
        """
        Test the newsvendor problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            # Classical Newsvendor Model
            # This model determines the optimal order quantity Q for a single-period stochastic inventory problem (the newsvendor problem).
            # The goal is to maximize expected profit, considering revenue from units sold, cost of ordering, and salvage value for surplus stock.

            # -------- Parameters --------
            param float revenue_per_unit;    # Revenue per unit sold
            param float cost_per_unit;       # Cost per unit ordered
            param float salvage_value;       # Salvage value per leftover unit

            tuple Scenario { int demand; float prob; }
            {Scenario} Scenarios = ...;      # Set of demand scenarios with probabilities

            float MaxDemand = max(s in Scenarios) (s.demand);

            # -------- Decision Variables --------
            dvar float+ Q;                    # Order quantity (continuous, nonnegative)

            # Number of units sold in each scenario: bounded above by both Q and realized demand.
            dvar float+ sold[Scenarios];      # Sold[s] = actual units sold in scenario s; 0 <= sold[s] <= min(Q, s.demand)

            # Number of leftover units in each scenario: Q - units sold, i.e., stock leftover after meeting demand (or 0 if all sold).
            dvar float+ leftover[Scenarios];  # leftover[s] = Q - sold[s]

            # -------- Objective Function --------
            # Maximizes the expected profit across all demand scenarios
            maximize expected_profit:
                sum(s in Scenarios) (
                    s.prob * (
                        revenue_per_unit * sold[s]
                    - cost_per_unit * Q
                    + salvage_value * leftover[s]
                    )
                );

            # -------- Constraints --------
            subject to {
                # Sales cannot exceed order quantity or scenario demand
                forall(s in Scenarios)
                    sold[s] <= Q;
                forall(s in Scenarios)
                    sold[s] <= s.demand;

                # leftover[s] = Q - sold[s]
                forall(s in Scenarios)
                    leftover[s] == Q - sold[s];

                # Upper bound on Q: practical maximum is the highest demand scenario
                Q <= MaxDemand;
            }
            """
        data_code = """
            # Instance parameters for the classical newsvendor problem
            revenue_per_unit = 10;
            cost_per_unit = 6;
            salvage_value = 2;

            # Demand scenarios: <demand, probability>
            Scenarios = { <400,0.1>, <600,0.2>, <700,0.4>, <800,0.2>, <1000,0.1> };
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_static_stochastic_knapsack(self):
        """
        Test the static stochastic knapsack problem with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Static Stochastic Knapsack (scenario-based, here-and-now decisions)
            //
            // Problem summary (literate notes):
            // - We must choose a subset of items before uncertainty realizes (static decision x[i] in {0,1}).
            // - Each scenario s in Scenarios has a probability p[s], and induces item weights w[s][i] and values v[s][i].
            // - The knapsack capacity C must be satisfied with probability at least 1 - epsilon (a chance constraint).
            // - Objective is to maximize the expected total value of selected items.
            //
            // Modeling constructs:
            // - Decision variables:
            //     x[i] in {0,1}: pick item i (here-and-now, scenario-independent).
            //     y[s] in {0,1}: indicator that scenario s respects capacity (1 if capacity holds under s).
            // - Parameters (inputs): capacity C, risk epsilon, scenario probabilities p, scenario weights w, scenario values v.
            // - Derived quantities:
            //     Ev[i] = sum_s p[s] * v[s][i] — expected value contribution of item i.
            //     BigM[s] = max(0, sum_i w[s][i] - C) — scenario-specific relaxation magnitude.
            //       When y[s] = 0, BigM[s] relaxes the capacity enough to never cut off any x.
            //
            // Objective (label: expected_value): maximize sum_i Ev[i] * x[i].
            // Constraints:
            //   - chance_requirement: sum_s p[s] * y[s] >= 1 - epsilon  (probability mass of feasible scenarios ≥ target)
            //   - capacity_linking (for all s): sum_i w[s][i] * x[i] <= C + BigM[s] * (1 - y[s])  (links y to capacity feasibility)

            // Index sets
            range Items = 1..5;                 // item indices (use numeric indices to align with array data)
            {string} Scenarios = ...;     // scenario identifiers (typed set of strings)

            // Parameters (external unless defined inline)
            param float C = ...;                         // knapsack capacity
            param float epsilon = ...;                   // risk tolerance in [0,1); probability of violation <= epsilon
            param float p[Scenarios] = ...;              // scenario probabilities (should sum to 1)
            param float w[Scenarios][Items] = ...;       // scenario-dependent weights (s,i)
            param float v[Scenarios][Items] = ...;       // scenario-dependent values (s,i)

            // Derived expected value per item: Ev[i] = E[v_i]
            dexpr float Ev[i in Items] = sum(s in Scenarios) p[s] * v[s][i];

            // Big-M per scenario to relax capacity when y[s] = 0
            // Use maxl to ensure nonnegativity while staying within supported function calls.
            param float BigM[s in Scenarios] = maxl( (sum(i in Items) w[s][i]) - C, 0 );

            // Decisions
            // x[i] = 1 if item i is selected; 0 otherwise (here-and-now)
            // y[s] = 1 if capacity is met in scenario s; 0 otherwise
            dvar boolean x[Items];
            dvar boolean y[Scenarios];

            // Objective: maximize expected total value of selected items
            maximize expected_value: sum(i in Items) Ev[i] * x[i];

            subject to {
            // Chance constraint: probability of meeting capacity >= 1 - epsilon
            chance_requirement: sum(s in Scenarios) p[s] * y[s] >= 1 - epsilon;

            // Scenario-wise capacity linking with big-M relaxation
            forall(s in Scenarios)
                capacity_linking: sum(i in Items) w[s][i] * x[i] <= C + BigM[s] * (1 - y[s]);
            }
            """
        data_code = """
            Scenarios = { "S1", "S2", "S3", "S4" };

            C = 9.0;
            epsilon = 0.25;  // require at least 75% probability mass to satisfy capacity

            // Scenario probabilities (sum to 1)
            p = [
            "S1" 0.25,
            "S2" 0.25,
            "S3" 0.25,
            "S4" 0.25
            ];

            // Scenario-dependent weights per scenario (rows) over Items = 1..5
            w = [
            "S1" [2.0, 5.0, 1.0, 1.0, 1.0],
            "S2" [2.2, 5.6, 1.2, 1.1, 1.0],
            "S3" [1.8, 5.1, 1.0, 1.2, 1.0],
            "S4" [2.5, 6.0, 1.5, 1.0, 1.2]
            ];

            // Scenario-dependent values per scenario (rows) over Items = 1..5
            v = [
            "S1" [9.0, 13.0, 5.0, 2.0, 1.5],
            "S2" [8.5, 12.5, 5.0, 2.0, 1.5],
            "S3" [9.0, 12.0, 4.5, 2.5, 1.2],
            "S4" [8.0, 13.5, 5.0, 1.8, 1.0]
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_p_dispersion(self):
        """
        Test the P-Dispersion Problem with both solvers.
        """
        model_code = """
            # P-Dispersion Problem (Kuby, 1987)
            int N = ...;
            range Sites = 1..N;
            int p = ...;
            float dist[Sites][Sites] = ...;

            # Decision variables
            dvar boolean y[Sites];
            dvar float+ z;

            # Upper bound for z (max inter-site distance)
            param float maxD = ...;

            maximize z;

            subject to {
            # Select exactly p sites
            sum(i in Sites) y[i] == p;

            # Bound z to aid linearization
            z <= maxD;

            # If both i and j are selected, z cannot exceed their separation
            forall(i in Sites, j in Sites : i < j)
                (y[i] + y[j] >= 2) => (z <= dist[i][j]);
            }
            """
        data_code = """
            N = 6;
            p = 3;
            maxD = 41;

            dist = [
            [0, 31, 22, 15, 28, 36],
            [31, 0, 27, 19, 33, 41],
            [22, 27, 0, 14, 25, 30],
            [15, 19, 14, 0, 18, 26],
            [28, 33, 25, 18, 0, 12],
            [36, 41, 30, 26, 12, 0]
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("gurobi",):  # Disable scipy for this test as it does not support this class of implications
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["gurobi"]["objective_value"],
            31.0,
            places=6,
        )

    def test_complex_workforce_planning_3(self):
        """
        Test a complex workforce planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // TRAINING AND PRODUCTION PLANNING (PyOPL)
            // Minimal revision: removed a single redundant weekly forbid (see comment below).
            // The rest of the model is unchanged; extensive comments and meaningful labels are kept
            // so the model acts as literate documentation of the decisions, parameters, objective,
            // and constraints.

            // TIME INDEX
            range Weeks = 1..8;

            // PARAMETERS (problem data)
            param int originalSkilled = 50;            // initial number of skilled workers (fixed)
            param float hours_normal = 40;             // normal hours per week per worker
            param float overtime_extra_hours = 20;     // extra hours when working 60h/week (60-40)
            param float rateI = 10;                    // kg per hour for product I
            param float rateII = 6;                    // kg per hour for product II
            param float wage_orig = 360;               // weekly wage of an original skilled worker
            param float wage_new = 240;                // weekly wage of a newly trained worker (after training)
            param float wage_trainee = 120;            // weekly wage of a trainee during the 2-week training
            param float wage_overtime = 540;           // weekly wage for a worker on overtime (60 h total)
            param float penI = 0.5;                    // penalty per kg per week delayed for product I
            param float penII = 0.6;                   // penalty per kg per week delayed for product II

            // Weekly demand (provided in the .dat file)
            param int demandI[Weeks] = ...;
            param int demandII[Weeks] = ...;

            // DECISION VARIABLES
            // startTrain[w]: number of new trainees who start training in week w
            // (a start in week s implies training during weeks s and s+1)
            dvar int+ startTrain[Weeks];

            // trainersCommitOrigStart[s] / trainersCommitNewStart[s]: number of trainers (origin) who
            // begin a 2-week trainer commitment in week s. These trainers are unavailable for production
            // during their 2-week commitment (weeks s and s+1).
            dvar int+ trainersCommitOrigStart[Weeks];
            dvar int+ trainersCommitNewStart[Weeks];

            // Overtime workers by origin (orig/new)
            // Note: the conservative variant below disallows overtime for newly trained workers.
            dvar int+ overtimeOrig[Weeks];
            dvar int+ overtimeNew[Weeks];

            // Production quantities (kg) for each product in each week
            dvar float+ producedI[Weeks];
            dvar float+ producedII[Weeks];

            // Backlog (unfulfilled demand) at end of each week for each product
            // Backlogs are nonnegative; each week's backlog is charged the per-week per-kg penalty
            dvar float+ backI[Weeks];
            dvar float+ backII[Weeks];

            // DERIVED QUANTITIES (decision-expressions) for clarity and constraints
            // trainersCommitStart[w]: total trainers starting commitment in week w
            dexpr int trainersCommitStart[w in Weeks] = trainersCommitOrigStart[w] + trainersCommitNewStart[w];
            // trainersActive[w]: trainers active in week w are those who started in w or in w-1 (2-week commitment)
            dexpr int trainersActive[w in Weeks] = trainersCommitStart[w] + ( (w>1) ? trainersCommitStart[w-1] : 0 );
            // trainers active by origin
            dexpr int trainersActiveOrig[w in Weeks] = trainersCommitOrigStart[w] + ( (w>1) ? trainersCommitOrigStart[w-1] : 0 );
            dexpr int trainersActiveNew[w in Weeks] = trainersCommitNewStart[w] + ( (w>1) ? trainersCommitNewStart[w-1] : 0 );

            // traineesInTraining[w]: trainees in training during week w (start in s are in s and s+1)
            dexpr int traineesInTraining[w in Weeks] = startTrain[w] + ( (w>1) ? startTrain[w-1] : 0 );

            // trainedSkilled[w]: number of new workers that have completed training and are available as skilled in week w
            // a trainee who started at s is available in weeks s+2 and onward
            dexpr int trainedSkilled[w in Weeks] = sum(s in Weeks : s <= w-2) startTrain[s];

            // overtimeTotal[w]: total overtime workers active in week w
            dexpr int overtimeTotal[w in Weeks] = overtimeOrig[w] + overtimeNew[w];

            // OBJECTIVE: minimize total cost
            // Break-down of objective terms (per week):
            //  - originalSkilled * wage_orig: base weekly wages paid to the original 50 skilled workers
            //  - trainedSkilled[w] * wage_new: weekly wages for newly-trained workers once they are available
            //  - overtimeOrig * (wage_overtime - wage_orig) : overtime premium for original workers who work 60h
            //  - overtimeNew  * (wage_overtime - wage_new)  : overtime premium for new skilled (if allowed)
            //  - traineesInTraining[w] * wage_trainee: wages paid to trainees during their 2-week training
            //  - backI[w] * penI + backII[w] * penII: backlog delay penalties charged each week per kg delayed
            minimize total_cost:
            sum(w in Weeks)
                (
                originalSkilled * wage_orig
                + trainedSkilled[w] * wage_new
                + overtimeOrig[w] * (wage_overtime - wage_orig)
                + overtimeNew[w]  * (wage_overtime - wage_new)
                + traineesInTraining[w] * wage_trainee
                + backI[w] * penI + backII[w] * penII
                );

            subject to {
            // ---------------------------------------------------------------------------
            // TRAINING COHORT CAPACITY
            // Each trainer who begins a 2-week commitment in week s can train up to 3 trainees
            // (trainersCommit*3 >= startTrain ensures capacity). We only enforce the start constraint
            // for s=1..7 because a start in week 8 would spill beyond the horizon.
            // ---------------------------------------------------------------------------
            forall(s in 1..7) training_capacity:
                (trainersCommitOrigStart[s] + trainersCommitNewStart[s]) * 3 >= startTrain[s];

            // Forbid starting cohorts in week 8 (they would spill beyond horizon)
            forbid_week8_starts: startTrain[8] == 0;
            // Keep the forbid for original trainers in week 8 (a trainer starting in week 8 would be active in week 9)
            forbid_week8_trainer_orig: trainersCommitOrigStart[8] == 0;
            // NOTE: removed forbid_week8_trainer_new (trainersCommitNewStart[8] == 0) because the
            // global constraint no_new_trainers (below) already forces trainersCommitNewStart[s] == 0
            // for all s. Removing the single-week ban avoids redundancy while keeping semantics identical.

            // New trainers who begin a commitment in week s must already be trained/available in week s
            // (they cannot be trainees in the same cohort they train)
            forall(s in Weeks) new_trainer_availability:
                trainersCommitNewStart[s] <= trainedSkilled[s];

            // Active trainers by origin cannot exceed available workers of that origin in that week
            forall(w in Weeks) trainer_availability_orig:
                trainersActiveOrig[w] <= originalSkilled;
            forall(w in Weeks) trainer_availability_new:
                trainersActiveNew[w] <= trainedSkilled[w];

            // ---------------------------------------------------------------------------
            // OVERTIME, AVAILABILITY, AND MUTUAL EXCLUSION
            // ---------------------------------------------------------------------------
            // Overtime workers cannot be trainers at the same time (for each origin):
            forall(w in Weeks) overtime_bound_orig:
                overtimeOrig[w] <= originalSkilled - trainersActiveOrig[w];
            forall(w in Weeks) overtime_bound_new:
                overtimeNew[w] <= trainedSkilled[w] - trainersActiveNew[w];

            // Total trainers active plus overtime workers cannot exceed total skilled workforce available that week
            forall(w in Weeks) availability:
                trainersActive[w] + overtimeTotal[w] <= originalSkilled + trainedSkilled[w];

            // ---------------------------------------------------------------------------
            // MINIMAL INTERPRETATION CHANGE (TEXTBOOK / CONSERVATIVE):
            // Disallow newly trained workers from being scheduled for overtime.
            // This enforces that only the ORIGINAL skilled workforce may be used for overtime
            // during the transition period. Remove this block if your interpretation allows
            // newly trained workers to take overtime immediately.
            // ---------------------------------------------------------------------------
            forall(w in Weeks) no_overtime_new:
                overtimeNew[w] == 0;

            // ---------------------------------------------------------------------------
            // NEW MINIMAL CHANGE (to match the alternative textbook assumption):
            // Forbid newly trained workers from serving as TRAINERS (conservative interpretation).
            // If the target solution assumed that only the ORIGINAL 50 can act as trainers,
            // enabling this constraint makes model semantics match that assumption.
            // Remove or comment out this block if newly trained workers should be allowed to train others.
            // ---------------------------------------------------------------------------
            forall(s in Weeks) no_new_trainers:
                trainersCommitNewStart[s] == 0;

            // ---------------------------------------------------------------------------
            // PRODUCTION CAPACITY (HOURS -> KG)
            // Producers are skilled workers not assigned as trainers. Each such worker provides
            // hours_normal hours; overtime workers add overtime_extra_hours.
            // Convert production (kg) into required hours by dividing by per-hour rates and
            // ensure capacity suffices.
            // ---------------------------------------------------------------------------
            forall(w in Weeks) production_capacity:
                ( producedI[w] / rateI + producedII[w] / rateII )
                <= ( (originalSkilled + trainedSkilled[w] - trainersActive[w]) * hours_normal ) + overtimeTotal[w] * overtime_extra_hours;

            // ---------------------------------------------------------------------------
            // BACKLOG (DELAY) BALANCE
            // backlog at end of week w = previous backlog + demand - production
            // (nonnegative by domain of backI/backII). This enforces that production cannot
            // exceed available demand + previous backlog (because back variables are >= 0).
            // ---------------------------------------------------------------------------
            forall(w in Weeks) backlog_I:
                backI[w] == ( (w>1) ? backI[w-1] : 0 ) + demandI[w] - producedI[w];
            forall(w in Weeks) backlog_II:
                backII[w] == ( (w>1) ? backII[w-1] : 0 ) + demandII[w] - producedII[w];

            // ---------------------------------------------------------------------------
            // TRAINING TARGET: by end of week 8 at least 50 new workers must have completed training
            // A trainee who starts in week s completes and becomes available in week s+2.
            // Therefore, to be available by the end of week 8, starts must occur no later than week 6.
            // ---------------------------------------------------------------------------
            training_goal:
                sum(s in 1..6) startTrain[s] >= 50;

            // ---------------------------------------------------------------------------
            // SIMPLE BOUNDS & CLARITY CONSTRAINTS (domains already enforced by dvar types)
            // ---------------------------------------------------------------------------
            forall(w in Weeks) nonneg_prod: producedI[w] >= 0;
            forall(w in Weeks) nonneg_prod2: producedII[w] >= 0;
            forall(w in Weeks) nonneg_back: backI[w] >= 0;
            forall(w in Weeks) nonneg_back2: backII[w] >= 0;
            }
            """
        data_code = """
            // Workforce Planning DSM - data file (example)
            demandI = [ 10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000 ];
            demandII = [ 6000, 7200, 8400, 10800, 10800, 12000, 12000, 12000 ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_complex_workforce_planning_2(self):
        """
        Test a complex workforce planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Workforce Planning DSM - model file (example)
            range T = 1..8;
            range TS = 1..7;

            param int S0 = ...;
            param float D1[T] = ...;  // demand for product I (kg)
            param float D2[T] = ...;  // demand for product II (kg)

            param float r1 = 10;      // kg per hour for product I
            param float r2 = 6;       // kg per hour for product II
            param int   h40 = 40;     // regular weekly hours
            param int   h60 = 60;     // overtime weekly hours

            param float w_s  = 360;   // weekly wage skilled (40h)
            param float w_new = 240;  // weekly wage new trained (40h)
            param float w_tr = 120;   // weekly wage trainee during training
            param float w_ot = 540;   // weekly wage overtime (60h) for any worker at 60h

            param float pen1 = 0.5;   // backlog penalty per kg per week for I
            param float pen2 = 0.6;   // backlog penalty per kg per week for II

            // Decisions
            // Training starts (cohorts start week s and last for two weeks: s and s+1)
            dvar int+ train[TS];      // number of trainees starting in week s
            dvar int+ trainer[TS];    // number of skilled trainers assigned to start in week s (each trains up to 3)

            // Workforce deployment per week
            dvar int+ xs[T];          // skilled producing at 40h in week t
            dvar int+ xso[T];         // skilled producing at 60h in week t
            dvar int+ xn[T];          // new trained producing at 40h in week t
            dvar int+ xno[T];         // new trained producing at 60h in week t

            // Production hour allocations and resulting production
            dvar float+ hI[T];
            dvar float+ hII[T];
            dvar float+ pI[T];
            dvar float+ pII[T];

            // Inventory and backlog
            dvar float+ sI[T];
            dvar float+ sII[T];
            dvar float+ bI[T];
            dvar float+ bII[T];

            // Derived expressions per week
            dexpr int trainersBusy[t in T] = sum(s in TS : (s == t) || (s + 1 == t)) trainer[s];
            dexpr int traineesBusy[t in T] = sum(s in TS : (s == t) || (s + 1 == t)) train[s];
            dexpr int newAvail[t in T]     = sum(s in TS : (s + 2) <= t) train[s];

            minimize totalCost:
            sum(t in T) (
                // wages for skilled (producing 40h or 60h) and trainers
                w_s * xs[t] + w_ot * xso[t] + w_s * trainersBusy[t]
                // wages for newly trained (40h or 60h)
                + w_new * xn[t] + w_ot * xno[t]
                // trainees during training (two weeks per start)
                + w_tr * traineesBusy[t]
                // backlog penalties
                + pen1 * bI[t] + pen2 * bII[t]
            );

            subject to {
            // Training capacity: each trainer handles up to 3 trainees per start (over 2 weeks)
            forall(s in TS) 3 * trainer[s] >= train[s];

            // Training goal: 50 new workers finished by end of week 8 (last start is week 7)
            sum(s in TS) train[s] == 50;

            // All initial skilled are either producing (40h or 60h) or training each week
            forall(t in T) xs[t] + xso[t] + trainersBusy[t] == S0;

            // All trained workers (available by week t) are assigned (40h or 60h)
            forall(t in T) xn[t] + xno[t] == newAvail[t];

            // Weekly production-hour capacity from assigned workers (skilled and trained)
            forall(t in T) hI[t] + hII[t] <= h40 * (xs[t] + xn[t]) + h60 * (xso[t] + xno[t]);

            // Production rates linking hours to output
            forall(t in T) {
                pI[t] == r1 * hI[t];
                pII[t] == r2 * hII[t];
            }

            // Inventory/backlog balance for product I
            (sI[1] - bI[1]) == 0 + pI[1] - D1[1];
            forall(t in 2..8) (sI[t] - bI[t]) == (sI[t-1] - bI[t-1]) + pI[t] - D1[t];

            // Inventory/backlog balance for product II
            (sII[1] - bII[1]) == 0 + pII[1] - D2[1];
            forall(t in 2..8) (sII[t] - bII[t]) == (sII[t-1] - bII[t-1]) + pII[t] - D2[t];
            }
            """
        data_code = """
            // Workforce Planning DSM - data file (example)
            S0 = 50;
            D1 = [10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000];
            D2 = [6000, 7200, 8400, 10800, 10800, 12000, 12000, 12000];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_complex_workforce_planning_1(self):
        """
        Test a complex workforce planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Workforce Planning DSM - model file (example)
            range Weeks = 1..8;
            {string} Food = {"I","II"};

            // Parameters
            float rate[Food] = [10, 6];
            int    Hn = 40;
            int    Ho = 60;
            float  wageS = 360;
            float  wageTrainee = 120;
            float  wageTrained = 240;
            float  wageOT = 540;
            int    S = 50;                 // legacy skilled (constant)
            int    target = 50;            // need 50 trained by end of week 8
            int    maxPerTrainer = 3;      // 3 trainees per trainer over 2 weeks
            float  pen[Food] = [0.5, 0.6];
            int    demand[Food][Weeks];    // fill with Table 1-11

            // Decision variables (per week)
            dvar int+ x[Weeks];            // new trainees starting in week t (force x[7]=x[8]=0)
            dvar int+ T1[Weeks];           // trainees in week-1 of training
            dvar int+ T2[Weeks];           // trainees in week-2 of training
            dvar int+ Z[Weeks];            // trainers (legacy skilled) engaged this week
            dvar int+ R[Weeks];            // trained workers available

            dvar int+ nS[Weeks];           // skilled working normal
            dvar int+ oS[Weeks];           // skilled working overtime
            dvar int+ trS[Weeks];          // skilled training (trainers)
            dvar int+ nR[Weeks];           // trained working normal
            dvar int+ oR[Weeks];           // trained working overtime

            dvar float+ prod[Food][Weeks];
            dvar float+ y[Food][Weeks][Weeks];  // allocation to order w in week t (t>=w)
            dvar float+ hours[Weeks];

            // Objective
            minimize
            // Labor costs (exactly once)
            sum(t in Weeks) (
                wageS*(nS[t] + trS[t]) + wageOT*oS[t]
            + wageTrained*nR[t]     + wageOT*oR[t]
            + wageTrainee*(T1[t] + T2[t])
            )
            // Late penalties with (t - w) factor
            + sum(f in Food, w in Weeks, t in w+1..8) ((t - w) * pen[f] * y[f][w][t]);

            // Workforce flow
            subject to{
            // Training pipeline
            forall(t in Weeks) {
                T1[t] == x[t];
                T2[t] == ((t>=2) ? x[t-1] : 0);
                R[t]  == ((t>=2) ? R[t-1] + ((t>=3) ? x[t-2] : 0) : 0);
            }
            // Trainer requirement
            forall(t in Weeks) {
                trS[t] >= Z[t];
                maxPerTrainer * Z[t] >= T1[t] + T2[t];
            }
            // Skilled balance by role
            forall(t in Weeks) nS[t] + oS[t] + trS[t] == S;
            // Trained balance by role
            forall(t in Weeks) nR[t] + oR[t] == R[t];

            // Training finish by week 8
            sum(t in 1..6) x[t] == target;   // last starts at week 6
            x[7] == 0; x[8] == 0;

            // Capacity

            forall(t in Weeks) {
                hours[t] == Hn*(nS[t]+nR[t]) + Ho*(oS[t]+oR[t]);
                sum(f in Food) prod[f][t] / rate[f] <= hours[t];
            }


            // Delivery flow and demand satisfaction
            // No early shipment and production link
            forall(f in Food, t in Weeks)
                (sum(w in 1..t) y[f][w][t]) <= prod[f][t];

            // Exact demand fulfillment (on time or late)
            forall(f in Food, w in Weeks)
                sum(t in w..8) y[f][w][t] == demand[f][w];
            }
            """
        data_code = """
            // Workforce Planning DSM - data file (example)
            demand = [
            [10000, 10000, 12000, 12000, 16000, 16000, 20000, 20000],
            [6000, 7200, 8400, 10800, 10800, 12000, 12000, 12000]
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        results = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                results[solver] = result
            finally:
                os.remove(model_file)
                os.remove(data_file)

        # If both solvers are infeasible, test passes
        if results["scipy"]["status"] == "INFEASIBLE" and results["gurobi"]["status"] == "INFEASIBLE":
            return  # Test passes

        # Otherwise, require both to be optimal and compare objectives
        self.assertEqual(results["scipy"]["status"], "OPTIMAL")
        self.assertEqual(results["gurobi"]["status"], "OPTIMAL")
        self.assertIn("objective_value", results["scipy"])
        self.assertIn("objective_value", results["gurobi"])
        self.assertAlmostEqual(
            results["scipy"]["objective_value"],
            results["gurobi"]["objective_value"],
            places=6,
        )

    def test_iterator_scoping_sum_and_forall(self):
        """
        Iterator scoping: ensure iterators introduced by sum(...) and forall(...) are
        in scope while parsing their bodies, including tuple-field index usage.

        Model constructs:
          - 3D boolean decision var x[I][J][K]
          - tuple T { int i; int j; int k; }
          - {T} Triples = { <i,j,k> | i in I, j in J, k in K };
          - Objective: maximize sum(c in Triples) x[c.i][c.j][c.k];
          - Constraint with nested iterators: forall(i in I, j in J) sum(k in K) x[i][j][k] >= 0;

        This test primarily validates parsing (no "Undeclared symbol" errors for i,j,k or c.i).
        """
        model_code = """
            int a = 2;
            int b = 2;
            int c = 2;
            range I = 1..a;
            range J = 1..b;
            range K = 1..c;

            tuple T { int i; int j; int k; }
            {T} Triples =
              { <i,j,k> | i in I, j in J, k in K };

            dvar boolean x[I][J][K];

            maximize sum(c in Triples) x[c.i][c.j][c.k];

            subject to {
              // Trivial feasibility constraint using forall with two iterators
              forall(i in I, j in J)
                sum(k in K) x[i][j][k] >= 0;
            }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"{solver} failed: {result.get('message')}")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
        # Cross-solver objective agreement
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_employee_rostering(self):
        """
        Test employee rostering model with both solvers.
        """
        model_code = """
            tuple Pair { string e; string s; }

            {string} Employees = ...;
            {string} Shifts = ...;

            {Pair} Pairs = { <e,s> | e in Employees, s in Shifts };

            int demand[Shifts];
            int pref[Pairs];

            dvar boolean x[Employees][Shifts];

            maximize sum(e in Employees, s in Shifts) pref[<e,s>] * x[e][s];

            subject to {
            forall (e in Employees)
                sum (s in Shifts) x[e][s] == 1;

            forall (s in Shifts)
                sum (e in Employees) x[e][s] == demand[s];
            }
            """
        data_code = """
            Employees = { "Alex", "Bri", "Casey", "Drew", "Evan" };
            Shifts = { "Morning", "Midday", "Evening" };

            demand = [ "Morning" 2, "Midday" 2, "Evening" 1 ];

            pref = [
            <"Alex","Morning"> 3, <"Alex","Midday"> 1, <"Alex","Evening"> 0,
            <"Bri","Morning"> 2, <"Bri","Midday"> 3, <"Bri","Evening"> 1,
            <"Casey","Morning"> 1, <"Casey","Midday"> 2, <"Casey","Evening"> 3,
            <"Drew","Morning"> 0, <"Drew","Midday"> 3, <"Drew","Evening"> 2,
            <"Evan","Morning"> 3, <"Evan","Midday"> 0, <"Evan","Evening"> 2
            ];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_minl_maxl_in_index_constraint(self):
        """
        Exercise minl/maxl inside forall index constraints with boolean AND.

        Ensures:
          - minl/maxl(pr.i, pr.i2) and minl/maxl(pr.j, pr.j2) used in the forall index filter
          - '&&' is parsed/evaluated in index constraints
        """
        model_code = """
            int a = 3;            // rows
            int b = 3;            // cols
            range Rows = 1..a;
            range Cols = 1..b;

            tuple Pair {
              int i;
              int j;
              int i2;
              int j2;
            }

            // Positive-area rectangles in row-major order
            {Pair} Pairs =
              { <i,j,i2,j2> |
                  i in Rows, j in Cols,
                  i2 in Rows, j2 in Cols :
                  // row-major strict ordering and positive area
                  ((i < i2) || (i == i2 && j < j2)) && (i != i2) && (j != j2)
              };

            dvar float+ y[Pairs];

            minimize
              sum(pr in Pairs) y[pr];

            subject to {
              // For each pr=(i,j,i2,j2), for each cell (m,n) strictly inside its rectangle,
              // add a trivial nonnegativity constraint using an index filter with minl/maxl and &&
              forall(pr in Pairs, m in Rows, n in Cols :
                (minl(pr.i, pr.i2) < m && m < maxl(pr.i, pr.i2)) &&
                (minl(pr.j, pr.j2) < n && n < maxl(pr.j, pr.j2))
              )
                y[pr] >= 0;
            }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED", f"{solver} failed: {result.get('message')}")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        # Objective should be zero for both (y >= 0 and minimized)
        self.assertAlmostEqual(obj_values["scipy"], 0.0, places=6)
        self.assertAlmostEqual(obj_values["gurobi"], 0.0, places=6)

    def test_tuple_set_comprehension_pairs(self):
        """
        Exercise tuple-set comprehension:
          {Pair} Pairs =
            { <i,j,i2,j2> |
                i in Rows, j in Cols,
                i2 in Rows, j2 in Cols :
                ((i-1)*b + j) < ((i2-1)*b + j2) // row-major strict ordering
            };
        Verifies that Pairs is materialized with the expected row-major ordering and size.
        """
        model_code = """
            int a = ...;          // rows
            int b = 3;            // cols
            range Rows = 1..a;
            range Cols = 1..b;

            tuple Pair {
              int i;
              int j;
              int i2;
              int j2;
            }

            {Pair} Pairs =
              { <i,j,i2,j2> |
                  i in Rows, j in Cols,
                  i2 in Rows, j2 in Cols :
                  ((i-1)*b + j) < ((i2-1)*b + j2)
              };

            // Trivial model using Pairs to ensure codegen touches it
            dvar boolean y[Pairs];

            minimize sum(pr in Pairs) y[pr];

            subject to {
              forall(pr in Pairs) y[pr] >= 0;
            }
            """
        data_code = """
            a = 2;
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_param_multi_index_rhs_expression_initialization(self):
        """
        Test model with multi-indexed parameter initialized from expression with both solvers.
        """
        model_code = """
            {int} I = ...;
            {int} J = {1, 2};

            // Multi-indexed parameter initialized from expression
            param float W[i in I][j in J] = i + j;
            param float X[i in I, j in J] = i + j; // Alternate syntax

            // Tie z to sum of W to exercise evaluation
            dvar float z;

            maximize z;
            subject to {
            z == sum(i in I, j in J) W[i][j];
            z == sum(i in I, j in J) X[i][j];
            }
            """
        data_code = """
            I = {1, 2, 3};
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_tuples_index_specifiers(self):
        """
        Test model with tuple index specifiers with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            tuple K { int i; string s; }
            {K} KS = ...;

            dvar float x[KS];

            maximize x[<1, "A">] + x[<2, "B">];

            subject to {
              // Make the problem bounded so maximize is well-defined
              x[<1, "A">] <= 1;
              x[<2, "B">] <= 1;

              // Non-negativity
              x[<1, "A">] >= 0;
              x[<2, "B">] >= 0;
            }
            """
        data_code = """
            KS = { <1, "A">, <2, "B"> };
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_min_max(self):
        """
        Test minl/maxl model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            int N = 4;
            float a[1..N] = ...;
            dvar float x[1..N];
            minimize max(i in 1..N) (a[i] * x[i]);
            subject to {
                sum(i in 1..N) x[i] == 1;
                forall(i in 1..N) x[i] >= 0;
                min(i in 1..N) (x[i]) >= 0.1;
            }
            """
        data_code = """
            a = [2.0, 4.5, 1.0, 3.0];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_minl_maxl(self):
        """
        Test minl/maxl model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            range N = 1..5;
            float a[N] = ...;
            dvar float x[N];
            dvar float y;

            minimize maxl(y, 0);

            subject to {
            sum(i in N) x[i] == 1;
            forall(i in N)
                x[i] >= 0;
            forall(i in N)
                y >= a[i] * x[i];
            min_constr: minl(x[1], x[2], x[3], x[4], x[5]) >= 0.1;
            max_constr: maxl(x[2], x[3], x[4], x[5]) <= 0.7;
            }
            """
        data_code = """
            a = [2.3, 4.7, 1.1, 3.5, 5.2];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_wagner_whitin_backorders(self):
        """
        Test Wagner-Whitin with backorders.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            param int nbPeriods = ...;
            range T = 1..nbPeriods;

            param float demand[T] = ...;
            param float setup_cost[T] = ...;
            param float prod_cost[T] = ...;
            param float hold_cost[T] = ...;
            param float penalty_cost[T] = ...;

            dvar float+ Q[T];          // quantity produced in period t
            dvar boolean y[T];         // setup binary: 1 if setup in period t
            dvar float I[T];           // inventory after period t (can be negative for backlog)
            dvar float+ h[T];          // max(s[t], 0): inventory held (aux variable)
            dvar float+ p[T];          // max(-s[t], 0): backlog/penalty (aux variable)

            minimize
            sum(t in T) ( setup_cost[t]*y[t]
                            + prod_cost[t]*Q[t]
                            + hold_cost[t]*h[t]
                            + penalty_cost[t]*p[t] );

            subject to {
            forall(t in T)
                if (t == 1) {
                balance_1: I[1] == Q[1] - demand[1];
                } else {
                balance_t: I[t] == I[t-1] + Q[t] - demand[t];
                }

            forall(t in T)
                prod_link: Q[t] <= y[t] * sum(k in t..nbPeriods) demand[k];

            forall(t in T)
                hold_lb: h[t] >= I[t];

            forall(t in T)
                penal_lb: p[t] >= -I[t];
            }
            """
        data_code = """
            nbPeriods = 6;
            demand = [20, 40, 30, 10, 50, 60];
            setup_cost = [100, 80, 100, 120, 110, 90];
            prod_cost = [5, 5, 5, 5, 5, 5];
            hold_cost = [1, 1, 1, 1, 1, 1];
            penalty_cost = [2, 2, 2, 2, 2, 2];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_production_planning_conditional_compare_solvers_1(self):
        """
        Test production planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            int N = 6;
            range T = 1..N;
            range TT = 2..N;
            dvar float+ Q[T];
            dvar float+ I[T];
            dvar boolean order[T];
            float demand[T] = ...;

            float K = 5;
            float h = 1;

            minimize sum(t in T) (K*order[t] + h*I[t]);
            subject to {
                forall(t in T){
                    if (t == 1) {
                        I[1] == Q[1] - demand[1];
                    } else {
                        I[t] == I[t-1] + Q[t] - demand[t];
                    }
                    // SciPy accepts inequality consequent; Q[t] is float+ so this forces Q[t]=0
                    (order[t] == 0) => (Q[t] <= 0);
                }
            }
            """
        data_code = """
            demand = [80, 60, 70, 90, 50, 60];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_production_planning_conditional_compare_solvers_2(self):
        """
        Test production planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            int N = 6;
            range T = 1..N;
            range TT = 2..N;
            dvar float+ Q[T];
            dvar float+ I[T];
            dvar boolean order[T];
            float demand[T] = ...;

            float K = 5;
            float h = 1;

            minimize sum(t in T) (K*order[t] + h*I[t]);
            subject to {
                forall(t in T){
                    if (t == 1) {
                        I[1] == Q[1] - demand[1];
                    } else {
                        I[t] == I[t-1] + Q[t] - demand[t];
                    }
                    // if produce, then setup
                    (Q[t] > 0) => (order[t] == 1);
                }
            }
            """
        data_code = """
            demand = [80, 60, 70, 90, 50, 60];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_production_planning_compare_solvers(self):
        """
        Test production planning model with both solvers.
        Checks that both solvers produce the same objective value for the given data.
        """
        model_code = """
            // Production Planning
            int nbProducts = ...;
            range Products = 1..nbProducts;
            int nbPeriods = ...;
            range Periods = 1..nbPeriods;
            float cost[Products][Periods] = ...;
            float demand[Periods] = ...;
            float capacity[Periods] = ...;

            dvar float+ x[Products][Periods];

            minimize sum(p in Products, t in Periods) cost[p][t] * x[p][t];

            subject to {
                forall(p in Products)
                    sum(t in Periods) x[p][t] >= demand[p];
                forall(t in Periods)
                    sum(p in Products) x[p][t] <= capacity[t];
            }
            """
        data_code = """
            nbProducts = 2;
            nbPeriods = 3;
            cost = [ [3, 2, 4], [2, 3, 5] ];
            demand = [40, 50, 0];
            capacity = [30, 40, 20];
            """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["scipy"], obj_values["gurobi"], places=6)

    def test_vehicle_routing_with_nested_tuples_dat(self):
        """
        Test vehicle routing problem with nested tuples, where arcs and nodes are read from a .dat file.
        Checks that both solvers return the same objective value.
        """
        model_code = """
        tuple Node {
            int id;
            float x;
            float y;
        };
        tuple Arc {
            Node from;
            Node to;
            float cost;
        };
        {Node} nodes = ...;
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) (x[a]) == 1;
        }
        """
        data_code = """
        nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >, < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >, < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        """
        obj_values = {}
        for solver in ("scipy", "gurobi"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_not_operator_in_forall_and_constraint(self):
        """TDD: Parser/codegen support for logical NOT '!' inside forall index constraint and implication.

        Uses: forall(i in 1..3 : !(i == 2)) x[i] >= 0; and (!(x[1] == 0)) => (x[1] >= 0);
        Should parse to AST containing 'not' nodes. Initially fails before implementation.
        """
        model_code = """
        range I = 1..3;
        dvar float x[I];
        dvar boolean y;
        minimize x[1];
        subject to {
            forall(i in I : !(i == 2)) x[i] >= 0;
            (!(x[1] == 0)) => (x[1] >= 0);
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        try:
            ast = parser.parse(lexer.tokenize(model_code))
        except Exception as e:
            raise e  # Ensure failing test prior to implementation

        def has_not(node):
            if isinstance(node, dict):
                if node.get("type") == "not":
                    return True
                return any(has_not(v) for v in node.values())
            if isinstance(node, list):
                return any(has_not(x) for x in node)
            return False

        self.assertTrue(has_not(ast), "Expected at least one 'not' node in AST for ! operator usage")

    def test_and_or_operators_in_constraint_and_implication(self):
        """TDD: Parser/codegen support for logical AND '&&' and OR '||' in constraints and implications.

        Model uses:
          (a == 1) && (b == 0);
          (a == 1) || (b == 1);
          (a == 1) && (b == 0) => y == 1;
          (a == 0) || (b == 1) => y == 0;
        Ensures AST contains 'and' and 'or' nodes. Gurobi codegen should succeed.
        """
        model_code = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean y;
        minimize a;
        subject to {
            (a == 1) && (b == 0);
            (a == 1) || (b == 1);
            (a == 1) && (b == 0) => y == 1;
            (a == 0) || (b == 1) => y == 0;
        }
        """
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model_code))
        found_and = False
        found_or = False

        def walk(node):
            nonlocal found_and, found_or
            if isinstance(node, dict):
                t = node.get("type")
                if t == "and":
                    found_and = True
                elif t == "or":
                    found_or = True
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(ast)
        self.assertTrue(found_and, "Expected at least one 'and' node in AST for && operator usage")
        self.assertTrue(found_or, "Expected at least one 'or' node in AST for || operator usage")
        # Gurobi code generation should include 'and'/'or' text (string form of Python boolean ops)
        code = GurobiCodeGenerator(ast).generate_code()
        self.assertIn(" and ", code)
        self.assertIn(" or ", code)

    def test_composite_boolean_implication(self):
        """Composite antecedent (a && b) => (c || !d) linearization with auxiliaries (Gurobi) and fallback (SciPy).
        Gurobi should build model; SciPy currently lacks composite boolean linearization and should raise.
        """
        model_code = """
        dvar boolean a;
        dvar boolean b;
        dvar boolean c;
        dvar boolean d;
        minimize a;
        subject to {
            (a == 1) && (b == 1) => (c == 1) || !(d == 1);
        }
        """
        # Parse once
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(model_code))
        # Ensure 'and' and 'or' and 'not' nodes present
        found = {k: False for k in ["and", "or", "not"]}

        def walk(n):
            if isinstance(n, dict):
                t = n.get("type")
                if t in found:
                    found[t] = True
                for v in n.values():
                    walk(v)
            elif isinstance(n, list):
                for x in n:
                    walk(x)

        walk(ast)
        for k, v in found.items():
            self.assertTrue(v, f"Missing {k} node in composite implication AST")
        # Gurobi codegen should succeed and contain implication aux constructs
        code = GurobiCodeGenerator(ast).generate_code()
        # We no longer rely on specific 'impl_bin' name; ensure auxiliary binary variables were introduced
        self.assertRegex(code, r"_b\d+_c0")
        # SciPy solve should raise (unsupported) for now
        import os
        import tempfile

        from pyopl.pyopl_core import solve_with_scipy

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
            tmp.write(model_code)
            tmp.flush()
            path = tmp.name
        try:
            res = solve_with_scipy(path)
            # SciPy currently unsupported for implication => expect FAILED status, message may be generic
            self.assertEqual(res["status"], "FAILED")
            msg = res.get("message", "")
            self.assertTrue("Implication constraints are not supported" in msg or "Failed to load or parse OPL model" in msg)
        finally:
            os.remove(path)

    def test_vehicle_routing_with_nested_tuples(self):
        """
        This test extends the vehicle routing problem with tuples by including nested tuples.
        It checks tuple type, set of nested tuples, dvar indexed by nested tuples, and constraints using nested tuple fields.
        It also checks that both solvers return the same objective value.
        """
        code = """
        tuple Node {
            int id;
            float x;
            float y;
        };
        tuple Arc {
            Node from;
            Node to;
            float cost;
        };
        {Node} nodes = { <1,0.0,0.0>, <2,1.0,0.0>, <3,0.0,1.0> };
        {Arc} arcs = { < <1,0.0,0.0>, <2,1.0,0.0>, 10.0 >, < <2,1.0,0.0>, <3,0.0,1.0>, 12.5 >, < <3,0.0,1.0>, <1,0.0,0.0>, 8.0 > };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(n in nodes)
                sum(a in arcs : a.from.id == n.id) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(n in nodes)
                sum(a in arcs : a.to.id == n.id) (x[a]) == 1;
        }
        """
        obj_values = {}
        for solver in ("scipy", "gurobi"):
            print(f"\n[DEBUG] Parsing with solver: {solver}")
            print("[DEBUG] OPL code being parsed:")
            print(code)
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Check tuple type declarations
            tuple_types = [d for d in ast["declarations"] if d["type"] == "tuple_type"]
            self.assertEqual(len(tuple_types), 2)
            self.assertEqual(tuple_types[0]["name"], "Node")
            self.assertEqual(tuple_types[1]["name"], "Arc")
            # Check set of tuples declaration
            set_of_nodes_decl = next((d for d in ast["declarations"] if d.get("name") == "nodes"), None)
            set_of_arcs_decl = next((d for d in ast["declarations"] if d.get("name") == "arcs"), None)
            self.assertIsNotNone(set_of_nodes_decl)
            self.assertIsNotNone(set_of_arcs_decl)
            # Check dvar indexed by arcs
            dvar_decl = next((d for d in ast["declarations"] if d.get("name") == "x"), None)
            self.assertIsNotNone(dvar_decl)
            self.assertEqual(dvar_decl["var_type"], "boolean")
            # Check objective is sum over arcs of a.cost * x[a]
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            # Check left side is a field_access (a.cost)
            left = sum_expr["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "cost")
            # Check right side is indexed_name (x[a])
            right = sum_expr["right"]
            self.assertEqual(right["type"], "indexed_name")
            self.assertEqual(right["name"], "x")
            # Check constraints: two forall constraints
            constraints = ast["constraints"]
            forall_constrs = [c for c in constraints if c["type"] == "forall_constraint"]
            self.assertEqual(len(forall_constrs), 2)
            # Check the first forall constraint structure (outgoing arcs)
            fc1 = forall_constrs[0]
            self.assertEqual(fc1["iterators"][0]["iterator"], "n")
            inner1 = fc1["constraint"]
            self.assertEqual(inner1["type"], "constraint")
            self.assertEqual(inner1["op"], "==")
            # The left side should be a sum with index constraint a.from.id == n.id
            left1 = inner1["left"]
            self.assertEqual(left1["type"], "sum")
            self.assertEqual(left1["index_constraint"]["type"], "binop")
            self.assertEqual(left1["index_constraint"]["op"], "==")
            # The right side should be 1
            self.assertEqual(inner1["right"]["type"], "number")
            self.assertEqual(inner1["right"]["value"], 1)
            # --- Solve the model and store the objective value ---
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                model_file = tmp.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        # Check that both solvers return the same objective value (within tolerance)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_wagner_whitin_linear(self):
        """
        Test Wagner-Whitin 5-period lot-sizing model with both solvers.
        Checks that both solvers produce the expected objective and solution for the Wagner-Whitin model with provided data.
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model (PyOPL syntax)

        int T = 5; // Number of periods
        float demand[1..T] = [20, 40, 30, 10, 50]; // Demand per period
        float unit_cost = 2;   // Unit production cost per period
        float setup_cost = 100; // Setup cost per period
        float holding_cost = 1; // Holding cost per period

        dvar float x[1..T]; // Amount produced in period t
        dvar float s[0..T]; // Inventory at end of period t
        dvar boolean y[1..T]; // 1 if setup/order occurs in period t

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= y[t] * sum(tt in t..T) demand[tt] ;
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
            y[1] == 1;
            y[2] == 0;
            y[3] == 0;
            y[4] == 0;
            y[5] == 1;
        }
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve, solve_with_scipy

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
            try:
                for solver, solve_fn in [
                    ("gurobi", solve),
                    ("scipy", solve_with_scipy),
                ]:
                    result = solve_fn(model_file)
                    self.assertNotEqual(result["status"], "FAILED")
                    self.assertIn("objective_value", result)
                    self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
                    sol = result.get("solution", {})
                    # Normalize variable names for comparison
                    norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
                    for k, v in expected_solution.items():
                        self.assertIn(k, norm_sol, f"Missing variable {k} in {solver} solution")
                        self.assertAlmostEqual(
                            norm_sol[k],
                            v,
                            places=4,
                            msg=f"{solver}: {k}={norm_sol[k]}, expected {v}",
                        )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)

    def test_wagner_whitin_model_data(self):
        """
        Test Wagner-Whitin 5-period lot-sizing model with both solvers.
        Checks that both solvers produce the expected objective and solution for the Wagner-Whitin model with provided data
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model (PyOPL syntax)

        int T = ...; // Number of periods
        float demand[1..T] = ...; // Demand per period
        float unit_cost = ...;    // Unit production cost per period
        float setup_cost = ...;   // Setup cost per period
        float holding_cost = ...; // Holding cost per period

        dvar float x[1..T]; // Amount produced in period t
        dvar float s[0..T]; // Inventory at end of period t
        dvar boolean y[1..T]; // 1 if setup/order occurs in period t

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            forall(t in 1..T)
                x[t] <= (sum(tt in t..T) demand[t]) * y[t];
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
        }
        """
        data_code = """
        // Wagner-Whitin 5-period lot-sizing model data (PyOPL syntax)
        T = 5; // Number of periods
        demand = [20, 40, 30, 10, 50]; // Demand per period
        unit_cost = 2;   // Unit production cost per period
        setup_cost = 100; // Setup cost per period
        holding_cost = 1; // Holding cost per period
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve, solve_with_scipy

        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
        ):
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
            tmp_dat.write(data_code)
            tmp_dat.flush()
            data_file = tmp_dat.name
            try:
                for solver, solve_fn in [
                    ("gurobi", solve),
                    ("scipy", solve_with_scipy),
                ]:
                    result = solve_fn(model_file, data_file)
                    self.assertNotEqual(result["status"], "FAILED")
                    self.assertIn("objective_value", result)
                    self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
                    sol = result.get("solution", {})
                    # Normalize variable names for comparison
                    norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
                    for k, v in expected_solution.items():
                        self.assertIn(k, norm_sol, f"Missing variable {k} in {solver} solution")
                        self.assertAlmostEqual(
                            norm_sol[k],
                            v,
                            places=4,
                            msg=f"{solver}: {k}={norm_sol[k]}, expected {v}",
                        )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)

    def test_wagner_whitin_implication(self):
        """
        Variant of Wagner-Whitin 5-period lot-sizing model using implication constraints:
        x[t] > 0 => y[t] == 1
        Should solve with Gurobi, and raise error with SciPy.
        """
        model_code = """
        // Wagner-Whitin 5-period lot-sizing model with implication constraint
        int T = 5;
        float demand[1..T] = [20, 40, 30, 10, 50];
        float unit_cost = 2;
        float setup_cost = 100;
        float holding_cost = 1;

        dvar float x[1..T];
        dvar float s[0..T];
        dvar boolean y[1..T];

        minimize
            sum(t in 1..T)
                (unit_cost * x[t] + setup_cost * y[t] + holding_cost * s[t]);

        subject to {
            s[0] == 0;
            forall(t in 1..T)
                x[t] + s[t-1] == demand[t] + s[t];
            //forall(t in 1..T)
            //    x[t] <= (sum(tt in t..T) demand[t]) * y[t];
            forall(t in 1..T) {
                x[t] >= 0;
                s[t] >= 0;
            }
            // Implication: if x[t] > 0 then y[t] == 1
            forall(t in 1..T)
                (x[t] > 0) => (y[t] == 1);
        }
        """
        expected_obj = 630.0
        expected_solution = {
            "x[1]": 100.0,
            "x[2]": 0.0,
            "x[3]": 0.0,
            "x[4]": 0.0,
            "x[5]": 50.0,
            "s[0]": 0.0,
            "s[1]": 80.0,
            "s[2]": 40.0,
            "s[3]": 10.0,
            "s[4]": 0.0,
            "s[5]": 0.0,
            "y[1]": 1.0,
            "y[2]": 0.0,
            "y[3]": 0.0,
            "y[4]": 0.0,
            "y[5]": 1.0,
        }
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
            tmp_mod.write(model_code)
            tmp_mod.flush()
            model_file = tmp_mod.name
        try:
            # Gurobi: should solve
            result = solve(model_file, solver="gurobi")
            self.assertNotEqual(result["status"], "FAILED")
            self.assertIn("objective_value", result)
            self.assertAlmostEqual(result["objective_value"], expected_obj, places=4)
            sol = result.get("solution", {})
            norm_sol = {self.normalize_varname(k): v for k, v in sol.items()}
            for k, v in expected_solution.items():
                self.assertIn(k, norm_sol, f"Missing variable {k} in gurobi solution")
                self.assertAlmostEqual(
                    norm_sol[k],
                    v,
                    places=4,
                    msg=f"gurobi: {k}={norm_sol[k]}, expected {v}",
                )
            # SciPy: now supports this implication via big-M gating (x <= M*y)
            result_scipy = solve(model_file, solver="scipy")
            self.assertNotEqual(result_scipy["status"], "FAILED")
            self.assertIn("objective_value", result_scipy)
            self.assertAlmostEqual(result_scipy["objective_value"], expected_obj, places=4)
            sol_scipy = result_scipy.get("solution", {})
            norm_sol_scipy = {self.normalize_varname(k): v for k, v in sol_scipy.items()}
            for k, v in expected_solution.items():
                self.assertIn(k, norm_sol_scipy, f"Missing variable {k} in scipy solution")
                self.assertAlmostEqual(
                    norm_sol_scipy[k],
                    v,
                    places=4,
                    msg=f"scipy: {k}={norm_sol_scipy[k]}, expected {v}",
                )
        finally:
            if os.path.exists(model_file):
                os.remove(model_file)

    def run_test_case_gurobi(self, opl_code):
        """Helper: Check Gurobi code generation for a given OPL model string."""
        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = GurobiCodeGenerator(ast)
        gurobi_code = generator.generate_code()
        # Just check that code generation does not raise
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(gurobi_code, str)

    def test_job_shop(self):
        """
        Warehouse Location Problem
        """
        model_code = """
        // Job Shop Scheduling Problem
        int nbJobs = ...;
        int nbMachines = ...;
        range Jobs = 1..nbJobs;
        range Machines = 1..nbMachines;
        int duration[Jobs][Machines] = ...;
        int M = 1000;

        dvar int+ start[Jobs][Machines];
        dvar boolean z[Jobs][Jobs][Machines];
        dvar int+ makespan;

        minimize makespan;

        subject to {
        // Each job must be processed on each machine in order
        forall(j in Jobs, m in Machines)
            start[j][m] >= 0;
        // No overlap on machines (simplified)
        forall(m in Machines)
            forall(j1 in Jobs, j2 in Jobs: j1 != j2){
            start[j1][m] + duration[j1][m] <=  start[j2][m] - 1 + M * z[j1][j2][m];
            start[j2][m] + duration[j2][m] <=  start[j1][m] - 1 + M * (1 - z[j1][j2][m]);
            }
        // Each job must be processed on each machine in order
        forall(j in Jobs, m in 1..nbMachines-1)
            start[j][m+1] >= start[j][m] + duration[j][m];
        // Makespan constraint
        forall(j in Jobs)
            makespan >= start[j][nbMachines] + duration[j][nbMachines];
        }
        """
        data_code = """
        nbJobs = 3;
        nbMachines = 2;
        duration = [
        [3, 2],   // Job 1: Machine 1 = 3, Machine 2 = 2
        [2, 4],   // Job 2: Machine 1 = 2, Machine 2 = 4
        [5, 1]    // Job 3: Machine 1 = 5, Machine 2 = 1
        ];
        """
        obj_values = {}
        import tempfile

        for solver in ("gurobi", "scipy"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_warehouse_location(self):
        """
        Warehouse Location Problem
        """
        model_code = """
        // Warehouse Location Problem
        int nbWarehouses = ...;
        int nbCustomers = ...;

        range Warehouses = 1..nbWarehouses;
        range Customers = 1..nbCustomers;

        float fixed_cost[Warehouses] = ...;
        float trans_cost[Warehouses][Customers] = ...;
        float demand[Customers] = ...;
        float capacity[Warehouses] = ...;

        dvar boolean y[Warehouses];
        dvar float+ x[Warehouses][Customers];

        minimize sum(i in Warehouses) fixed_cost[i] * y[i] + sum(i in Warehouses, j in Customers) trans_cost[i][j] * x[i][j];

        subject to {
        forall(j in Customers)
            sum(i in Warehouses) x[i][j] == demand[j];
        forall(i in Warehouses, j in Customers)
            x[i][j] <= capacity[i] * y[i];
        }
        """
        data_code = """
        nbWarehouses = 2;
        nbCustomers = 3;
        fixed_cost = [80, 90];
        trans_cost = [ [3, 5, 8],
                       [4, 3, 6] ];
        demand = [15, 20, 10];
        capacity = [25, 30];
        """
        obj_values = {}
        import tempfile

        for solver in ("gurobi", "scipy"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_graph_coloring_tuples(self):
        """
        Graph Coloring Problem using tuples and sets.
        """
        model_code = """
        // Proper Graph Coloring Problem (no !=, uses big-M encoding)
        int nbNodes = ...;
        range Nodes = 1..nbNodes;

        tuple Edge {
            int source;
            int dest;
        };

        {Edge} Edges = ...;

        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Edges]; // auxiliary binary for big-M encoding

        minimize maxColor;

        subject to {
            // Each node's color is at least 1 and at most nbNodes
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            // Adjacent nodes must have different colors (big-M encoding)
            forall(e in Edges)
                color[e.source] >= color[e.dest] + 1 - nbNodes * z[e];
            forall(e in Edges)
                color[e.dest] >= color[e.source] + 1 - nbNodes * (1 - z[e]);
            // maxColor is at least as large as any color used
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        Edges = { <1,2>, <2,3>, <3,4>, <4,1> };
        """
        obj_values = {}
        import tempfile

        for solver in ("gurobi", "scipy"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_graph_coloring_matrix(self):
        """
        Graph Coloring Problem using an adjacency matrix (no tuples).
        """
        model_code = """
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        int adj[Nodes][Nodes] = ...; // adjacency matrix: 1 if edge, 0 otherwise

        dvar int+ color[Nodes];
        dvar int+ maxColor;
        dvar boolean z[Nodes][Nodes]; // auxiliary binary for big-M encoding

        minimize maxColor;

        subject to {
            // Each node's color is at least 1 and at most nbNodes
            forall(i in Nodes) color[i] >= 1;
            forall(i in Nodes) color[i] <= nbNodes;
            // Adjacent nodes must have different colors
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[i] >= color[j] + 1 - nbNodes * z[i][j];
            forall(i in Nodes, j in Nodes : adj[i][j] == 1)
                color[j] >= color[i] + 1 - nbNodes * (1-z[i][j]);
            // maxColor is at least as large as any color used
            forall(i in Nodes) maxColor >= color[i];
        }
        """
        data_code = """
        nbNodes = 4;
        adj = [
            [0,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
            [1,0,1,0]
        ];
        """
        obj_values = {}
        import tempfile

        for solver in ("gurobi", "scipy"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                print(f"\n[DEBUG] Solver: {solver}, Result: {result}")
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_matrix_dat(self):
        """
        Test vehicle routing problem with matrix, where arcs are read from a .dat file.
        """

        model_code = """
        // Matrix-based vehicle routing problem
        int nbNodes = ...;
        range Nodes = 1..nbNodes;
        float cost[Nodes][Nodes] = ...;
        dvar boolean x[Nodes][Nodes];
        minimize sum(i in Nodes, j in Nodes) cost[i][j] * x[i][j];
        subject to {
        forall(i in Nodes)
            sum(j in Nodes) (x[i][j]) == 1;
        forall(j in Nodes)
            sum(i in Nodes) (x[i][j]) == 1;
        }
        """
        data_code = """
        nbNodes = 3;
        cost = [
            [1000, 10.0, 1000],
            [1000, 1000, 12.5],
            [8.0, 1000, 1000]
        ];
        """
        obj_values = {}
        import tempfile

        for solver in ("scipy", "gurobi"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_with_tuples_dat(self):
        """
        Test vehicle routing problem with tuples, where arcs are read from a .dat file.
        """

        model_code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = ...;
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            forall(i in 1..3)
                sum(a in arcs : a.from == i) (x[a]) == 1;
            forall(j in 1..3)
                sum(a in arcs : a.to == j) (x[a]) == 1;
        }
        """
        data_code = """
        arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        """
        obj_values = {}
        import tempfile

        for solver in ("scipy", "gurobi"):

            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_vehicle_routing_with_tuples(self):
        """
        This test embeds a small vehicle routing problem using tuples, similar to classical OPL models.
        It checks tuple type, set of tuples, dvar indexed by tuples, and constraints using tuple fields.
        It also checks that both solvers return the same objective value.
        """
        code = """
        tuple Arc {
            int from;
            int to;
            float cost;
        };
        {Arc} arcs = { <1,2,10.0>, <2,3,12.5>, <3,1,8.0> };
        dvar boolean x[arcs];
        minimize sum(a in arcs) a.cost * x[a];
        subject to {
            // Each node must have exactly one outgoing arc
            forall(i in 1..3)
                sum(a in arcs : a.from == i) (x[a]) == 1;
            // Each node must have exactly one incoming arc
            forall(j in 1..3)
                sum(a in arcs : a.to == j) (x[a]) == 1;
        }
        """
        import os
        import tempfile

        obj_values = {}
        for solver in ("scipy", "gurobi"):
            print(f"\n[DEBUG] Parsing with solver: {solver}")
            print("[DEBUG] OPL code being parsed:")
            print(code)
            compiler = OPLCompiler()
            ast, code_str, data_dict = compiler.compile_model(code, solver=solver)
            # Check tuple type declaration
            tuple_type_decl = next((d for d in ast["declarations"] if d["type"] == "tuple_type"), None)
            self.assertIsNotNone(tuple_type_decl)
            self.assertEqual(tuple_type_decl["name"], "Arc")
            # Check set of tuples declaration
            set_of_tuples_decl = next((d for d in ast["declarations"] if d.get("name") == "arcs"), None)
            self.assertIsNotNone(set_of_tuples_decl)
            # Check dvar indexed by arcs
            dvar_decl = next((d for d in ast["declarations"] if d.get("name") == "x"), None)
            self.assertIsNotNone(dvar_decl)
            self.assertEqual(dvar_decl["var_type"], "boolean")
            # Check objective is sum over arcs of a.cost * x[a]
            obj = ast["objective"]
            self.assertEqual(obj["type"], "minimize")
            expr = obj["expression"]
            self.assertEqual(expr["type"], "sum")
            sum_expr = expr["expression"]
            self.assertEqual(sum_expr["type"], "binop")
            # Check left side is a field_access (a.cost)
            left = sum_expr["left"]
            self.assertEqual(left["type"], "field_access")
            self.assertEqual(left["field"], "cost")
            # Check right side is indexed_name (x[a])
            right = sum_expr["right"]
            self.assertEqual(right["type"], "indexed_name")
            self.assertEqual(right["name"], "x")
            # Check constraints: two forall constraints
            constraints = ast["constraints"]
            forall_constrs = [c for c in constraints if c["type"] == "forall_constraint"]
            self.assertEqual(len(forall_constrs), 2)
            # Check the first forall constraint structure (outgoing arcs)
            fc1 = forall_constrs[0]
            self.assertEqual(fc1["iterators"][0]["iterator"], "i")
            inner1 = fc1["constraint"]
            self.assertEqual(inner1["type"], "constraint")
            self.assertEqual(inner1["op"], "==")
            # The left side should be a sum with index constraint a.from == i
            left1 = inner1["left"]
            self.assertEqual(left1["type"], "sum")
            self.assertEqual(left1["index_constraint"]["type"], "binop")
            self.assertEqual(left1["index_constraint"]["op"], "==")
            # The right side should be 1
            self.assertEqual(inner1["right"]["type"], "number")
            self.assertEqual(inner1["right"]["value"], 1)
            # --- Solve the model and store the objective value ---
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(code)
                tmp.flush()
                model_file = tmp.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        # Check that both solvers return the same objective value (within tolerance)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_basic_production_planning_gurobi(self):
        """Test Gurobi codegen for a basic production planning model."""
        opl_code = """
        dvar float x;
        dvar float y;

        maximize x + y;

        subject to {
            x <= 10;
            y <= 15;
            x + y <= 20;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def run_test_case_scipy(self, opl_code, data_dict=None):
        """Helper: Check SciPy code generation for a given OPL model string."""
        from pyopl.pyopl_core import OPLLexer, OPLParser, SciPyCodeGenerator

        lexer = OPLLexer()
        parser = OPLParser()
        ast = parser.parse(lexer.tokenize(opl_code))
        generator = SciPyCodeGenerator(ast, data_dict or {})
        scipy_code = generator.generate_code()
        self.assertIsInstance(ast, dict)
        self.assertIsInstance(scipy_code, str)

    def test_basic_production_planning_scipy(self):
        """Test SciPy codegen for a basic production planning model."""
        opl_code = """
        dvar float x;
        dvar float y;

        maximize x + y;

        subject to {
            x <= 10;
            y <= 15;
            x + y <= 20;
        }
        """
        self.run_test_case_scipy(opl_code)

    def pyopl_vs_cplex_output(self, model, data, cplex_obj=None):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsack.mod/dat."""

        # Solve with pyopl (default solver is gurobi)
        result = pyopl.solve(model, data)
        self.assertNotIn(result["status"], ["ERROR", "FAILED", "EXECUTION_ERROR"])
        if isinstance(result, dict) and "objective_value" in result:
            gurobi_obj = result["objective_value"]
        else:
            gurobi_obj = result
        if cplex_obj is not None:
            self.assertAlmostEqual(
                gurobi_obj,
                cplex_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+gurobi: {gurobi_obj}",
            )

        # Solve with pyopl using scipy
        result_scipy = pyopl.solve(model, data, solver="scipy")
        self.assertNotIn(result_scipy["status"], ["ERROR", "FAILED", "EXECUTION_ERROR"])
        if isinstance(result_scipy, dict) and "objective_value" in result_scipy:
            scipy_obj = result_scipy["objective_value"]
        else:
            scipy_obj = result_scipy
        if cplex_obj is not None:
            self.assertAlmostEqual(
                scipy_obj,
                cplex_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+scipy: {scipy_obj}",
            )

        if cplex_obj is None:
            self.assertAlmostEqual(
                scipy_obj,
                gurobi_obj,
                places=4,
                msg=f"CPLEX: {cplex_obj}, pyopl+scipy: {scipy_obj}",
            )

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_knapsack_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsack.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 10.0

        KNAPSACK_MOD = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsack.mod")
        KNAPSACK_DAT = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsack.dat")

        self.pyopl_vs_cplex_output(KNAPSACK_MOD, KNAPSACK_DAT, cplex_obj)

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_knapsackp_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsackp.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 498.0

        KNAPSACKP_MOD = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsackp.mod")
        KNAPSACKP_DAT = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/knapsack/knapsackp.dat")

        self.pyopl_vs_cplex_output(KNAPSACKP_MOD, KNAPSACKP_DAT, cplex_obj)

    @unittest.skipUnless(GUROBI_AVAILABLE, "pyopl or gurobi not available")
    def test_inventory_routing_pyopl_vs_cplex_output(self):
        """Compare pyopl (Gurobi/SciPy) solution to CPLEX reference for knapsackp.mod/dat."""
        # CPLEX reference solution
        cplex_obj = 103.0

        INVENTORY_ROUTING_MOD = os.path.join(
            os.path.dirname(__file__),
            "../pyopl/opl_models/inventory_routing/inventory_routing.mod",
        )
        INVENTORY_ROUTING_DAT = os.path.join(
            os.path.dirname(__file__),
            "../pyopl/opl_models/inventory_routing/inventory_routing.dat",
        )

        self.pyopl_vs_cplex_output(INVENTORY_ROUTING_MOD, INVENTORY_ROUTING_DAT, cplex_obj)

    def test_tsp_model_parsing_and_codegen_gurobi(self):
        """Test parsing and codegen for the TSP model (Gurobi)."""
        # Paths to the TSP model and data
        model_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/tsp/tsp.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/tsp/tsp.dat")
        with open(model_path) as f:
            model_code = f.read()
        with open(data_path) as f:
            data_code = f.read()
        compiler = OPLCompiler()
        ast, gurobi_code, data_dict = compiler.compile_model(model_code, data_code, solver="gurobi")
        print("\n==== DEBUG: Generated Gurobi Code ====")
        print(gurobi_code)
        print("==== END DEBUG ====")

        def find_node_with_index_constraint(node, node_type):
            if isinstance(node, dict):
                if node.get("type") == node_type and node.get("index_constraint") is not None:
                    return True
                # Recursively search all dict/list children
                for v in node.values():
                    if find_node_with_index_constraint(v, node_type):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_node_with_index_constraint(item, node_type):
                        return True
            return False

        found_sum = find_node_with_index_constraint(ast, "sum")
        found_forall = find_node_with_index_constraint(ast, "forall_constraint")
        self.assertTrue(found_sum, "Sum with index constraint not found in AST")
        self.assertTrue(found_forall, "Forall with index constraint not found in AST")
        # Check that the generated code uses itertools.product and 'if' for index constraint
        self.assertIn("itertools.product", gurobi_code)
        self.assertIn("if ", gurobi_code)
        self.assertIn("gp.quicksum", gurobi_code)
        # Optionally, check that the code compiles
        compile(gurobi_code, "<string>", "exec")

    def test_tsp_model_parsing_and_codegen_scipy(self):
        """Test parsing and codegen for the TSP model (SciPy)."""
        # Paths to the TSP model and data
        model_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/tsp/tsp.mod")
        data_path = os.path.join(os.path.dirname(__file__), "../pyopl/opl_models/tsp/tsp.dat")
        with open(model_path) as f:
            model_code = f.read()
        with open(data_path) as f:
            data_code = f.read()
        compiler = OPLCompiler()
        ast, scipy_code, data_dict = compiler.compile_model(model_code, data_code, solver="scipy")
        print("\n==== DEBUG: Generated SciPy Code ====")
        print(scipy_code)
        print("==== END DEBUG ====")

        def find_node_with_index_constraint(node, node_type):
            if isinstance(node, dict):
                if node.get("type") == node_type and node.get("index_constraint") is not None:
                    return True
                for v in node.values():
                    if find_node_with_index_constraint(v, node_type):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if find_node_with_index_constraint(item, node_type):
                        return True
            return False

        found_sum = find_node_with_index_constraint(ast, "sum")
        found_forall = find_node_with_index_constraint(ast, "forall_constraint")
        self.assertTrue(found_sum, "Sum with index constraint not found in AST (scipy)")
        self.assertTrue(found_forall, "Forall with index constraint not found in AST (scipy)")
        self.assertIn("linprog", scipy_code)
        self.assertIn("if ", scipy_code)
        # Optionally, check that the code compiles
        compile(scipy_code, "<string>", "exec")

    def test_knapsack_problem_compare_solvers(self):
        """Compare Gurobi and SciPy solutions for a generated knapsack problem."""
        from pyopl.pyopl_core import solve, solve_with_scipy

        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """)
            # Gurobi
            result_gurobi = solve(dummy_model_file, dummy_data_file)
            # SciPy
            result_scipy = solve_with_scipy(dummy_model_file, dummy_data_file)
            # Print diagnostic output
            print("Gurobi solution:", result_gurobi.get("solution", {}))
            print("Gurobi objective:", result_gurobi.get("objective_value"))
            print("SciPy solution:", result_scipy.get("solution", {}))
            print("SciPy objective:", result_scipy.get("objective_value"))
            # Only compare objectives, since multiple optima are possible
            try:
                self.compare_objectives(
                    result_gurobi.get("objective_value"),
                    result_scipy.get("objective_value"),
                )
            except AssertionError as e:
                msg = (
                    f"Objective mismatch in knapsack_problem_compare_solvers.\n"
                    f"Gurobi objective: {result_gurobi.get('objective_value')}\n"
                    f"SciPy objective: {result_scipy.get('objective_value')}\n"
                    f"Gurobi solution: {result_gurobi.get('solution', {})}\n"
                    f"SciPy solution: {result_scipy.get('solution', {})}\n"
                )
                raise AssertionError(msg) from e
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_knapsack_problem_scipy(self):
        """Test SciPy codegen and solution for a generated knapsack problem."""
        from pyopl.pyopl_core import load_opl_model, solve_with_scipy

        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """)
            ast, scipy_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(scipy_code, str)
            self.assertIsInstance(data_dict, dict)
            # Also test solve_with_scipy
            result = solve_with_scipy(dummy_model_file, dummy_data_file)
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def normalize_varname(self, name):
        import re

        # Accept x_1, x_1_2, x[1], x[1,2] and map all to canonical form x[1] or x[1,2]
        # Match var_1_2_3 -> var[1,2,3]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)((?:_[0-9]+)+)$", name)
        if m:
            indices = m.group(2).lstrip("_").split("_")
            return f"{m.group(1)}[{','.join(indices)}]"
        # Match var[1,2,3] or var[1]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)\[([0-9,]+)\]$", name)
        if m:
            return f"{m.group(1)}[{m.group(2)}]"
        # Match var_1 -> var[1]
        m = re.match(r"([a-zA-Z][a-zA-Z0-9]*)_([0-9]+)$", name)
        if m:
            return f"{m.group(1)}[{m.group(2)}]"
        return name

    def compare_solutions(self, sol1, sol2, tol=1e-5):
        # Normalize variable names in both solutions
        norm1 = {self.normalize_varname(k): v for k, v in sol1.items()}
        norm2 = {self.normalize_varname(k): v for k, v in sol2.items()}
        self.assertEqual(set(norm1.keys()), set(norm2.keys()))
        for k in norm1:
            self.assertAlmostEqual(norm1[k], norm2[k], delta=tol)

    def compare_objectives(self, obj1, obj2, tol=1e-5):
        self.assertAlmostEqual(obj1, obj2, delta=tol)

    def test_assignment_problem_compare_solvers(self):
        """Compare Gurobi and SciPy solutions for a generated assignment problem."""
        from pyopl.pyopl_core import solve, solve_with_scipy

        dummy_model_file = "assign_model.mod"
        dummy_data_file = "assign_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                dvar boolean assign[1..2][1..2];
                range Persons = 1..2;
                range Tasks = 1..2;

                minimize sum (p in Persons) (sum (t in Tasks) (5 * assign[p][t]));

                subject to {
                    forall (p in Persons)
                        sum (t in Tasks) (assign[p][t]) == 1;
                    forall (t in Tasks)
                        sum (p in Persons) (assign[p][t]) == 1;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("")  # No data needed
            # Gurobi
            result_gurobi = solve(dummy_model_file, dummy_data_file)
            # SciPy
            result_scipy = solve_with_scipy(dummy_model_file, dummy_data_file)
            # Compare solutions
            try:
                self.compare_solutions(result_gurobi.get("solution", {}), result_scipy.get("solution", {}))
            except AssertionError as e:
                msg = (
                    f"Solution mismatch in assignment_problem_compare_solvers.\n"
                    f"Gurobi solution: {result_gurobi.get('solution', {})}\n"
                    f"SciPy solution: {result_scipy.get('solution', {})}\n"
                )
                raise AssertionError(msg) from e
            try:
                self.compare_objectives(
                    result_gurobi.get("objective_value"),
                    result_scipy.get("objective_value"),
                )
            except AssertionError as e:
                msg = (
                    f"Objective mismatch in assignment_problem_compare_solvers.\n"
                    f"Gurobi objective: {result_gurobi.get('objective_value')}\n"
                    f"SciPy objective: {result_scipy.get('objective_value')}\n"
                )
                raise AssertionError(msg) from e
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_knapsack_problem(self):
        """Test Gurobi codegen and parsing for a generated knapsack problem."""
        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                range Items = 1..5;
                param float weight[1..5];
                param float value[1..5];
                param float C;

                dvar boolean x[1..5];

                maximize sum (i in Items) (value[i] * x[i]);

                subject to {
                    sum (i in Items) (weight[i] * x[i]) <= C;
                }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                weight = [2,3,4,5,5];
                value = [2,3,4,5,5];
                C = 10;
                """)
            ast, gurobi_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(gurobi_code, str)
            self.assertIsInstance(data_dict, dict)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_multi_resource_knapsack_problem(self):
        """Test Gurobi codegen and parsing for a multi-resource knapsack problem."""
        dummy_model_file = "knapsack_model.mod"
        dummy_data_file = "knapsack_data.dat"
        try:
            with open(dummy_model_file, "w") as f:
                f.write("""
                        range Items = 1..12;
                        range Resources = 1..7;
                        float Capacity[Items] = ...;
                        float Value[Items];
                        float Use[Resources][Items];

                        dvar boolean Take[Items];

                        maximize sum(i in Items) Value[i] * Take[i];

                        subject to {
                        forall( r in Resources )
                            sum( i in Items )
                                Use[r][i] * Take[i] <= Capacity[r];
                        }
                """)
            with open(dummy_data_file, "w") as f:
                f.write("""
                        Capacity = [ 18209, 7692, 1333, 924, 26638, 61188, 13360,
                                     18209, 7692, 1333, 924, 26638 ];
                        Value = [ 96, 76, 56, 11, 86, 10, 66, 86, 83, 12, 9, 81 ];
                        Use = [ [ 19,   1,  10,  1,   1,  14, 152, 11,  1,   1, 1, 1 ],
                            [  0,   4,  53,  0,   0,  80,   0,  4,  5,   0, 0, 0 ],
                            [  4, 660,   3,  0,  30,   0,   3,  0,  4,  90, 0, 0],
                            [  7,   0,  18,  6, 770, 330,   7,  0,  0,   6, 0, 0],
                            [  0,  20,   0,  4,  52,   3,   0,  0,  0,   5, 4, 0],
                            [  0,   0,  40, 70,   4,  63,   0,  0, 60,   0, 4, 0],
                            [  0,  32,   0,  0,   0,   5,   0,  3,  0, 660, 0, 9]];
                """)
            ast, gurobi_code, data_dict = load_opl_model(dummy_model_file, dummy_data_file)
            self.assertIsInstance(ast, dict)
            self.assertIsInstance(gurobi_code, str)
            self.assertIsInstance(data_dict, dict)
        finally:
            if os.path.exists(dummy_model_file):
                os.remove(dummy_model_file)
            if os.path.exists(dummy_data_file):
                os.remove(dummy_data_file)

    def test_transportation_problem(self):
        """Test Gurobi codegen for a transportation problem."""
        opl_code = """
        dvar float flow[1..2][1..3];
        range Origins = 1..2;
        range Destinations = 1..3;

        minimize sum (i in Origins) (sum (j in Destinations) (10 * flow[i][j]));

        subject to {
            forall (i in Origins)
                sum (j in Destinations) (flow[i][j]) <= 100;
            forall (j in Destinations)
                sum (i in Origins) (flow[i][j]) >= 50;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def test_simple_assignment_problem(self):
        """Test Gurobi codegen for a simple assignment problem."""
        opl_code = """
        dvar boolean assign[1..2][1..2];
        range Persons = 1..2;
        range Tasks = 1..2;

        minimize sum (p in Persons) (sum (t in Tasks) (5 * assign[p][t]));

        subject to {
            forall (p in Persons)
                sum (t in Tasks) (assign[p][t]) == 1;
            forall (t in Tasks)
                sum (p in Persons) (assign[p][t]) == 1;
        }
        """
        self.run_test_case_gurobi(opl_code)

    def test_multi_indexed_variable_and_constraint(self):
        """Test 3D indexed variables and constraints (multi-indexed arrays) with both solvers."""
        opl_code = """
        dvar float+ x[1..2][1..3][1..2];
        range I = 1..2;
        range J = 1..3;
        range K = 1..2;
        minimize sum(i in I, j in J, k in K) x[i][j][k];
        subject to {
            forall(i in I, j in J)
                sum(k in K) x[i][j][k] <= 5;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_tuple_field_access_and_nested_tuple_set(self):
        """Test tuple field access and nested tuple sets with both solvers."""
        opl_code = """
        tuple Inner { int id; float val; };
        tuple Outer { Inner inner; float weight; };
        {Outer} outers = { < <1, 2.5>, 3.0 >, < <2, 4.0>, 1.5 > };
        dvar float+ x[outers];
        minimize sum(o in outers) o.inner.val * x[o];
        subject to {
            forall(o in outers) x[o] <= o.weight;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_inline_and_external_data_mix(self):
        """Test model with both inline and .dat data, including parameter arrays."""
        model_code = """
        int N = ...;
        range I = 1..N;
        float cost[I] = ...;
        dvar float x[I];
        minimize sum(i in 1..N) cost[i] * x[i];
        subject to {
            forall(i in I) x[i] >= 0;
            sum(i in I) x[i] == 10;
        }
        """
        data_code = """
        N = 3;
        cost = [2.0, 3.0, 1.5];
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_filtered_sum_and_nested_forall(self):
        """Test constraints using filtered sums and nested forall with both solvers."""
        opl_code = """
        range I = 1..3;
        range J = 1..3;
        dvar boolean x[I][J];
        minimize sum(i in I, j in J) x[i][j];
        subject to {
            forall(i in I)
                sum(j in J : j != i) x[i][j] == 1;
            forall(j in J)
                sum(i in I : i != j) x[i][j] == 1;
        }
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
        self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_simple_blending_problem(self):
        """Test codegen & solve for a simple blending problem with both Gurobi and SciPy."""
        opl_code = """
        dvar float blendA;
        dvar float blendB;

        minimize 2.5 * blendA + 3.0 * blendB;

        subject to {
            blendA + blendB == 100;
            0.3 * blendA + 0.6 * blendB >=  45;
            0.1 * blendA + 0.2 * blendB <= 20;
        }
        """
        # Expected optimal solution: solve small LP analytically.
        # Binding constraints: blendA + blendB == 100 and 0.3A + 0.6B == 45 -> A + 2B = 150 -> A = 150 - 2B.
        # Substitute into A+B=100 -> (150 - 2B) + B = 100 -> 150 - B = 100 -> B = 50, A = 50.
        # Check third: 0.1*50 + 0.2*50 = 15 <= 20 OK. Objective = 2.5*50 + 3.0*50 = 125 + 150 = 275.
        expected_obj = 275.0
        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod:
                tmp_mod.write(opl_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
            try:
                result = solve(model_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=5,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_blending_string_sets_list_index_error(self):
        """Blending with string-indexed scalar sets and list data (bug regression test).

        Original bug: parameters stored as Python lists were indexed by string labels, raising
        'list indices must be integers or slices, not str'. The codegen now emits <Set>_index
        maps and remaps string labels to integer positions for both Gurobi and SciPy backends.

        This test verifies both solvers produce the same optimal objective (342.5) and thus
        guards against regressions in typed string set indexing for 1D/2D list parameters.
        """
        model_code = """
            {string} Products = ...;
            {string} Resources = ...;

            float Consumption[Products][Resources] = ...;
            float Capacity[Resources] = ...;
            float Demand[Products] = ...;
            float InsideCost[Products] = ...;
            float OutsideCost[Products]  = ...;

            dvar float+ Inside[Products];
            dvar float+ Outside[Products];

            minimize
                sum( p in Products )
                    ( InsideCost[p] * Inside[p] + OutsideCost[p] * Outside[p] );

            subject to {
                forall( r in Resources )
                    ctCapacity:
                        sum( p in Products )
                            Consumption[p][r] * Inside[p] <= Capacity[r];

                forall(p in Products)
                    ctDemand:
                        Inside[p] + Outside[p] >= Demand[p];
            }
            """
        data_code = """
            Products = { "ProdA", "ProdB" };
            Resources = { "Res1", "Res2" };

            Consumption = [
                [ 1.0, 2.0 ],
                [ 0.5, 1.5 ]
            ];
            Capacity = [ 100.0, 80.0 ];
            Demand = [ 40.0, 50.0 ];
            InsideCost = [ 2.0, 3.0 ];
            OutsideCost = [ 5.0, 6.0 ];
            """
        expected_obj = 342.5
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                # Objective close to expected
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_workforce_planning_conditional_vs_explicit(self):
        """
        Test that explicit and conditional-expression workforce planning models produce the same solution and objective.
        """
        # Explicit model (no conditional expressions)
        explicit_model = """
        // ----------------------
        // SETS AND PARAMETERS
        // ----------------------

        int T = ...; // Number of periods
        int S = ...; // Number of skill levels
        int K = ...; // Number of tasks (job types)

        range Periods = 1..T;
        range Skills = 1..S;
        range SkillTrans = 1..S-1; // Transitions for possible training from s to s+1
        range Tasks = 1..K;

        float hiringCost[Skills];
        float firingCost[Skills];
        float wage[Skills];
        float otWage[Skills];
        float productivity[Skills];
        float maxOvertime[Skills];
        float trainingCost[SkillTrans];

        int initialWorkforce[Skills];
        int demand[Tasks][Periods];
        int skillsRequired[Tasks][Skills];
        float budget[Periods];
        int maxHire[Skills][Periods];
        int maxFire[Skills][Periods];
        int spanControl;
        int nManagers;

        dvar int+ hire[Skills][Periods];
        dvar int+ fire[Skills][Periods];
        dvar int+ train[SkillTrans][Periods];
        dvar int+ assign[Skills][Tasks][Periods];
        dvar int+ overtime[Skills][Periods];
        dvar int+ workforce[Skills][Periods];

        minimize
        sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
        + sum(s in SkillTrans, p in Periods) trainingCost[s] * train[s][p]
        + sum(s in Skills, p in Periods) wage[s] * sum(t in Tasks) assign[s][t][p]
        + sum(s in Skills, p in Periods) otWage[s] * overtime[s][p];

        subject to {
            workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
            forall(s in 2..S)
            workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

            forall(p in 2..T)
            workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

            forall(s in 2..S-1, p in 2..T)
            workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

            forall(p in 2..T)
            workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

            forall(s in Skills, p in Periods)
            sum(t in Tasks) assign[s][t][p] <= workforce[s][p]*productivity[s] + overtime[s][p];

            forall(s in Skills, p in Periods)
            overtime[s][p] <= workforce[s][p]*maxOvertime[s];

            forall(s in Skills, p in Periods)
            hire[s][p] <= maxHire[s][p];
            forall(s in Skills, p in Periods)
            fire[s][p] <= maxFire[s][p];

            forall(s in Skills)
            fire[s][1] <= initialWorkforce[s];
            forall(s in Skills, p in 2..T)
            fire[s][p] <= workforce[s][p-1];

            forall(s in SkillTrans)
            train[s][1] <= initialWorkforce[s];
            forall(s in SkillTrans, p in 2..T)
            train[s][p] <= workforce[s][p-1];

            forall(t in Tasks, p in Periods)
            sum(s in Skills : skillsRequired[t][s]==1) assign[s][t][p] >= demand[t][p];

            forall(p in Periods)
            sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

            forall(p in Periods)
                sum(s in Skills)
                (hiringCost[s]*hire[s][p] + firingCost[s]*fire[s][p] + wage[s]*sum(t in Tasks) assign[s][t][p] + otWage[s]*overtime[s][p])
                + sum(s in SkillTrans)
                trainingCost[s]*train[s][p]
                <= budget[p];
        }
        """

        # Conditional-expression model
        conditional_model = """
        // ASSUMPTIONS:
        // * Time is discretized into periods.
        // * There is a finite and known set of skill levels and tasks.
        // * Productivity is normalized per worker per period.
        // * Overtime is allowed only up to a specified maximum per worker.
        // * All monetary values (costs, wages) and worker-hours are known input data.

        int T = ...;
        int S = ...;
        int K = ...;

        range Periods = 1..T;
        range Skills = 1..S;
        range SkillTrans = 1..S-1;
        range Tasks = 1..K;

        float hiringCost[Skills];
        float firingCost[Skills];
        float trainingCost[SkillTrans];
        float wage[Skills];
        float otWage[Skills];
        float productivity[Skills];
        float maxOvertime[Skills];
        int initialWorkforce[Skills];
        int demand[Tasks][Periods];
        int skillsRequired[Tasks][Skills];
        float budget[Periods];
        int maxHire[Skills][Periods];
        int maxFire[Skills][Periods];
        int spanControl;
        int nManagers;

        dvar int+ hire[Skills][Periods];
        dvar int+ fire[Skills][Periods];
        dvar int+ train[SkillTrans][Periods];
        dvar int+ assign[Skills][Tasks][Periods];
        dvar int+ overtime[Skills][Periods];
        dvar int+ workforce[Skills][Periods];

        minimize
        sum(s in Skills, p in Periods) (hiringCost[s] * hire[s][p] + firingCost[s] * fire[s][p])
        + sum(s in SkillTrans, p in Periods) trainingCost[s] * train[s][p]
        + sum(s in Skills, p in Periods) wage[s] * sum(t in Tasks) assign[s][t][p]
        + sum(s in Skills, p in Periods) otWage[s] * overtime[s][p];

        subject to {
            workforce[1][1] == initialWorkforce[1] + hire[1][1] - fire[1][1];
            forall(s in 2..S)
                workforce[s][1] == initialWorkforce[s] + hire[s][1] - fire[s][1];

            forall(p in 2..T)
                workforce[1][p] == workforce[1][p-1] + hire[1][p] - fire[1][p] - train[1][p-1];

            forall(s in 2..S-1, p in 2..T)
                workforce[s][p] == workforce[s][p-1] + hire[s][p] - fire[s][p] + train[s-1][p-1] - train[s][p-1];

            forall(p in 2..T)
                workforce[S][p] == workforce[S][p-1] + hire[S][p] - fire[S][p] + train[S-1][p-1];

            forall(s in Skills, p in Periods)
                sum(t in Tasks) assign[s][t][p] <= workforce[s][p]*productivity[s] + overtime[s][p];

            forall(s in Skills, p in Periods)
                overtime[s][p] <= workforce[s][p]*maxOvertime[s];

            forall(s in Skills, p in Periods)
                hire[s][p] <= maxHire[s][p];
            forall(s in Skills, p in Periods)
                fire[s][p] <= maxFire[s][p];

            forall(s in Skills)
                fire[s][1] <= initialWorkforce[s];
            forall(s in Skills, p in 2..T)
                fire[s][p] <= workforce[s][p-1];

            forall(s in SkillTrans)
                train[s][1] <= initialWorkforce[s];
            forall(s in SkillTrans, p in 2..T)
                train[s][p] <= workforce[s][p-1];

            forall(t in Tasks, p in Periods)
                sum(s in Skills : skillsRequired[t][s]==1) assign[s][t][p] >= demand[t][p];

            forall(p in Periods)
                sum(s in Skills) workforce[s][p] <= nManagers * spanControl;

            forall(p in Periods)
                sum(s in Skills)
                (hiringCost[s]*hire[s][p] + firingCost[s]*fire[s][p] + wage[s]*sum(t in Tasks) assign[s][t][p] + otWage[s]*overtime[s][p])
                + sum(s in SkillTrans)
                trainingCost[s]*train[s][p]
                <= budget[p];
        }
        """

        # Data file as provided
        data_code = """
        T = 3;    // number of periods
        S = 2;    // number of skill levels
        K = 2;    // number of tasks/job types

        hiringCost = [ 1000, 1500 ];
        firingCost = [ 500, 800 ];
        trainingCost = [ 700 ];   // only S-1, i.e., training from level 1 to level 2

        wage = [ 25, 35 ];
        otWage = [ 40, 55 ];

        productivity = [ 40, 50 ];

        maxOvertime = [ 10, 15 ];

        initialWorkforce = [ 15, 10 ];

        demand = [
        [ 400, 530, 460 ],
        [ 250, 220, 300 ]
        ];

        skillsRequired = [
        [1, 1],
        [0, 1]
        ];

        budget = [ 25000, 25000, 25000 ];

        maxHire = [
        [ 5, 5, 5 ],
        [ 3, 3, 3 ]
        ];
        maxFire = [
        [ 5, 5, 5 ],
        [ 3, 3, 3 ]
        ];

        spanControl = 10;
        nManagers = 3;
        """

        import os
        import tempfile

        from pyopl.pyopl_core import solve

        # Write models and data to temp files
        with (
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod1,
            tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod2,
            tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
        ):
            tmp_mod1.write(explicit_model)
            tmp_mod1.flush()
            tmp_mod2.write(conditional_model)
            tmp_mod2.flush()
            tmp_dat.write(data_code)
            tmp_dat.flush()
            model_file1 = tmp_mod1.name
            model_file2 = tmp_mod2.name
            data_file = tmp_dat.name

        try:
            # Test both solvers for both models
            for solver in ("gurobi", "scipy"):
                result_explicit = solve(model_file1, data_file, solver=solver)
                result_conditional = solve(model_file2, data_file, solver=solver)
                self.assertNotEqual(
                    result_explicit["status"],
                    "FAILED",
                    f"Explicit model failed for {solver}",
                )
                self.assertNotEqual(
                    result_conditional["status"],
                    "FAILED",
                    f"Conditional model failed for {solver}",
                )
                self.assertIn("objective_value", result_explicit)
                self.assertIn("objective_value", result_conditional)
                # Compare objective values
                self.assertAlmostEqual(
                    result_explicit["objective_value"],
                    result_conditional["objective_value"],
                    places=4,
                    msg=f"Objective mismatch for {solver}: explicit={result_explicit['objective_value']}, conditional={result_conditional['objective_value']}",
                )
                # Compare solutions (variable values)
                sol_explicit = result_explicit.get("solution", {})
                sol_conditional = result_conditional.get("solution", {})
                norm_explicit = {self.normalize_varname(k): v for k, v in sol_explicit.items()}
                norm_conditional = {self.normalize_varname(k): v for k, v in sol_conditional.items()}
                self.assertEqual(
                    set(norm_explicit.keys()),
                    set(norm_conditional.keys()),
                    msg=f"Variable set mismatch for {solver}: explicit={set(norm_explicit.keys())}, conditional={set(norm_conditional.keys())}",
                )
                for k in norm_explicit:
                    self.assertAlmostEqual(
                        norm_explicit[k],
                        norm_conditional[k],
                        places=4,
                        msg=f"Variable {k} mismatch for {solver}: explicit={norm_explicit[k]}, conditional={norm_conditional[k]}",
                    )
        finally:
            if os.path.exists(model_file1):
                os.remove(model_file1)
            if os.path.exists(model_file2):
                os.remove(model_file2)
            if os.path.exists(data_file):
                os.remove(data_file)

    def test_rich_opl_model(self):
        """
        Test a rich OPL model with ranges, tuples, sets, dvars, constraints, and data.
        Logical constraints are omitted due to lack of implementation.
        """
        model_code = """
        int N = ...;
        range Items = 1..N;

        tuple Product {
            int id;
            float profit;
            float weight;
        };

        {Product} products = ...;

        float capacity = ...;

        dvar boolean take[products];

        maximize sum(p in products) p.profit * take[p];

        subject to {
            sum(p in products) p.weight * take[p] <= capacity;
            forall(p in products){
                 //(take[p] == 0) || (take[p] == 1); //no general logical OR over linear constraints
                 (take[p]) + (1 - (take[p])) == 1;
            }
        }
        """
        data_code = """
        N = 4;
        products = { <1, 10.0, 2.0>, <2, 15.0, 3.0>, <3, 7.0, 1.5>, <4, 8.0, 2.5> };
        capacity = 5.0;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                tmp_dat.write(data_code)
                tmp_dat.flush()
                model_file = tmp_mod.name
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result["status"], "FAILED")
                self.assertIn("objective_value", result)
                obj_values[solver] = result["objective_value"]
            finally:
                os.remove(model_file)
                os.remove(data_file)

    def test_mini_graph_coloring_with_neq_and_implication(self):
        """4-cycle coloring using native != plus an extra linear implication for span-based big-M.

                Constructs used:
                    - color[i] != color[j] (numeric != with tightening)
                    - Implication: (color[1] == 2) => (color[2] >= 2)  (uses a simple equality antecedent supported by SciPy)
        Both backends should solve with same objective (minimize maxColor).
        """
        base_model = """
        int N = 4;
        range V = 1..N;
        tuple Edge { int u; int v; };
        {Edge} arcs = { <1,2>, <2,3>, <3,4>, <4,1> };
        dvar int+ color[V];
        dvar int+ maxColor;
        minimize maxColor;
        subject to {
            forall(i in V) {
                color[i] >= 1;
                color[i] <= 4;
                maxColor >= color[i];
            }
            // Edge coloring constraints via tuples
            forall(e in arcs)
                color[e.u] != color[e.v];
            // IMPL_LINE
        }
        """
        results = {}
        # Use implication with equality antecedent so SciPy can encode (pattern b==1 style).
        for solver in ("gurobi", "scipy"):
            model_code = base_model.replace("// IMPL_LINE", "(color[1] == 2) => (color[2] >= 2);")
            with tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp:
                tmp.write(model_code)
                tmp.flush()
                path = tmp.name
            try:
                res = solve(path, solver=solver)
                self.assertNotEqual(res.get("status"), "FAILED")
                if res.get("objective_value") is not None:
                    results[solver] = res.get("objective_value")
            finally:
                if os.path.exists(path):
                    os.remove(path)
        self.assertAlmostEqual(results["gurobi"], results["scipy"], places=6)

    def test_food_blending_problem(self):
        """Food blending problem: mixes ingredients into foods meeting nutrient demands.

        Model uses:
            - {string} sets Foods, Ingredients
            - tuple types foodType / ingredientType with demand, price, protein, fat fields
            - Arrays Food[Foods], Ingredient[Ingredients]
            - Decision vars: slack[Foods] (over-production), Mix[Ingredients][Foods]
            Objective maximizes margin minus slack penalties; optimal slack expected zero.
        """
        model_code = """
        {string} Foods = ...;
        {string} Ingredients = ...;
        tuple foodType { float demand; float price; float protein; float fat; };
        tuple ingredientType { float capacity; float price; float protein; float fat; };
        foodType Food[Foods] = ...;
        ingredientType Ingredient[Ingredients] = ...;
        float MaxProduction = ...;
        float ProcCost = ...; // processing cost per unit

        dvar float+ slack[Foods];
        dvar float+ Mix[Ingredients][Foods];

        maximize
            sum( f in Foods , ing in Ingredients )
                (Food[f].price - Ingredient[ing].price - ProcCost) * Mix[ing][f]
                - sum(f in Foods) slack[f];
        subject to {
            forall( f in Foods )
                sum( ing in Ingredients ) Mix[ing][f] == Food[f].demand + 10*slack[f];
            // Ingredient capacity
            forall( ing in Ingredients )
                sum( f in Foods ) Mix[ing][f] <= Ingredient[ing].capacity;
            // Global production limit
            sum( ing in Ingredients , f in Foods ) Mix[ing][f] <= MaxProduction;
            // Protein quality: blended protein must not fall below required (weighted diff >= 0)
            forall( f in Foods )
                sum( ing in Ingredients ) (Ingredient[ing].protein - Food[f].protein) * Mix[ing][f] >= 0;
            // Fat limit: blended fat must not exceed target (weighted diff <= 0)
            forall( f in Foods )
                sum( ing in Ingredients ) (Ingredient[ing].fat - Food[f].fat) * Mix[ing][f] <= 0;
        }
        """
        data_code = """
        Foods = { "Meal1", "Meal2", "Meal3" };
        Ingredients = { "Chicken", "Beef", "Soy" };

        Food = [ <3000, 9, 30, 10>,
                 <2000, 7, 25, 15>,
                 <1000, 6, 20, 12> ];

        Ingredient = [ <5000, 4, 35, 6>,
                        <5000, 5, 28, 18>,
                        <5000, 3, 22, 14> ];

        MaxProduction = 14000;
        ProcCost = 1.5;
        """
        expected_obj = 35766.66666666666
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
                # Objective close to expected
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_transportation_problem_with_tuples_and_string_sets(self):
        """
        Transportation Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Transportation Problem with advanced OPL constructs

        tuple Arc {
            string origin;
            string dest;
        }

        // Sets
        {string} Origins = ...;
        {string} Destinations = ...;
        {string} SpecialOrigins = { "Seattle" };
        {Arc} arcs = ...;

        // Parameters
        param float supply[Origins];
        param float demand[Destinations];
        param float cost[arcs];
        param float capacity[arcs];         // NEW: arc capacities
        param float min_shipment[arcs];     // NEW: minimum shipment per arc
        param float total_shipment_limit;   // NEW: global shipment limit

        // Decision variables
        dvar float+ x[arcs];

        // Objective
        minimize sum(a in arcs) cost[a] * x[a];

        subject to {
            // Supply constraints
            forall(o in Origins)
                sum(a in arcs : a.origin == o) x[a] <= supply[o];

            // Demand constraints
            forall(d in Destinations)
                sum(a in arcs : a.dest == d) x[a] >= demand[d];

            // Arc capacity constraints
            forall(a in arcs)
                x[a] <= capacity[a];

            // Minimum shipment on certain arcs
            forall(o in SpecialOrigins)
                forall(a in arcs : a.origin == o)
                    x[a] >= 10;

            // Total shipment limit
            sum(a in arcs) x[a] <= total_shipment_limit;
        }

        """
        data_code = """
        // Data for the transportation problem

        Origins = { "Seattle", "San-Diego" };
        Destinations = { "New-York", "Chicago", "Topeka" };

        arcs = {
            <"Seattle", "New-York">,
            <"Seattle", "Chicago">,
            <"Seattle", "Topeka">,
            <"San-Diego", "New-York">,
            <"San-Diego", "Chicago">,
            <"San-Diego", "Topeka">
        };

        supply = [
            "Seattle"   350,
            "San-Diego" 600
        ];

        demand = [
            "New-York" 325,
            "Chicago"  300,
            "Topeka"   275
        ];

        cost = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      2.5,
            <"Seattle", "Topeka">       1.7,
            <"San-Diego", "New-York">   2.5,
            <"San-Diego", "Chicago">    1.8,
            <"San-Diego", "Topeka">     1.4
        ];

        // Arc capacities
        capacity = [
            <"Seattle", "New-York">     200,
            <"Seattle", "Chicago">      250,
            <"Seattle", "Topeka">       200,
            <"San-Diego", "New-York">   300,
            <"San-Diego", "Chicago">    300,
            <"San-Diego", "Topeka">     400
        ];

        // Minimum shipment per arc (0 for most, but you can set >0 for some)
        min_shipment = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      0,
            <"Seattle", "Topeka">       0,
            <"San-Diego", "New-York">   0,
            <"San-Diego", "Chicago">    0,
            <"San-Diego", "Topeka">     50
        ];

        // Total shipment limit
        total_shipment_limit = 900;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_transportation_problem_with_tuples_and_string_sets_and_string_filtering(
        self,
    ):
        """
        Transportation Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Transportation Problem with advanced OPL constructs

        tuple Arc {
            string origin;
            string dest;
        }

        // Sets
        {string} Origins = ...;
        {string} Destinations = ...;
        {Arc} arcs = ...;

        // Parameters
        param float supply[Origins];
        param float demand[Destinations];
        param float cost[arcs];
        param float capacity[arcs];         // NEW: arc capacities
        param float min_shipment[arcs];     // NEW: minimum shipment per arc
        param float total_shipment_limit;   // NEW: global shipment limit

        // Decision variables
        dvar float+ x[arcs];

        // Objective
        minimize sum(a in arcs) cost[a] * x[a];

        subject to {
            // Supply constraints
            forall(o in Origins)
                sum(a in arcs : a.origin == o) x[a] <= supply[o];

            // Demand constraints
            forall(d in Destinations)
                sum(a in arcs : a.dest == d) x[a] >= demand[d];

            // Arc capacity constraints
            forall(a in arcs)
                x[a] <= capacity[a];

            // Minimum shipment on certain arcs
            forall(a in arcs : a.origin == "Seattle")
                x[a] >= 10;

            // Total shipment limit
            sum(a in arcs) x[a] <= total_shipment_limit;
        }

        """
        data_code = """
        // Data for the transportation problem

        Origins = { "Seattle", "San-Diego" };
        Destinations = { "New-York", "Chicago", "Topeka" };

        arcs = {
            <"Seattle", "New-York">,
            <"Seattle", "Chicago">,
            <"Seattle", "Topeka">,
            <"San-Diego", "New-York">,
            <"San-Diego", "Chicago">,
            <"San-Diego", "Topeka">
        };

        supply = [
            "Seattle"   350,
            "San-Diego" 600
        ];

        demand = [
            "New-York" 325,
            "Chicago"  300,
            "Topeka"   275
        ];

        cost = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      2.5,
            <"Seattle", "Topeka">       1.7,
            <"San-Diego", "New-York">   2.5,
            <"San-Diego", "Chicago">    1.8,
            <"San-Diego", "Topeka">     1.4
        ];

        // Arc capacities
        capacity = [
            <"Seattle", "New-York">     200,
            <"Seattle", "Chicago">      250,
            <"Seattle", "Topeka">       200,
            <"San-Diego", "New-York">   300,
            <"San-Diego", "Chicago">    300,
            <"San-Diego", "Topeka">     400
        ];

        // Minimum shipment per arc (0 for most, but you can set >0 for some)
        min_shipment = [
            <"Seattle", "New-York">     0,
            <"Seattle", "Chicago">      0,
            <"Seattle", "Topeka">       0,
            <"San-Diego", "New-York">   0,
            <"San-Diego", "Chicago">    0,
            <"San-Diego", "Topeka">     50
        ];

        // Total shipment limit
        total_shipment_limit = 900;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_inventory_problem_with_tuples(self):
        """
        Inventory Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Inventory Problem with tuple arcs and string-indexed sets.

        tuple Store { string name; }
        {Store} Stores;

        range Periods = 1..3;

        int Capacity[Stores];
        int Demand[Periods];
        int OrderingCost[Periods];
        int HoldingCost;

        dvar int I[Stores][Periods];
        dvar int Q[Stores][Periods];

        minimize sum(s in Stores, p in Periods) (OrderingCost[p] * Q[s][p] + HoldingCost * I[s][p]);

        subject to {
            forall(s in Stores)
                I[s][1] == Q[s][1] - Demand[1];
            forall(s in Stores, p in 2..3)
                I[s][p] == I[s][p-1] + Q[s][p] - Demand[p];

            forall(s in Stores, p in Periods)
                I[s][p] <= Capacity[s];

            forall(s in Stores, p in Periods)
                I[s][p] >= 0;

            forall(s in Stores, p in Periods)
                Q[s][p] >= 0;
        }
        """
        data_code = """
        // Data for the inventory problem

        Stores = { <"S1">, <"S2"> };
        Capacity = [<"S1"> 100, <"S2"> 100];
        Demand = [
            1, 2, 3
        ];
        OrderingCost = [10, 13 , 15];
        HoldingCost = 1;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        expected_obj = 136
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                print(f"{solver} objective: {obj}")
                self.assertAlmostEqual(
                    obj,
                    expected_obj,
                    places=3,
                    msg=f"{solver} objective {obj} != expected {expected_obj}",
                )
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        # Cross-solver objective agreement
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=3)

    def test_complex_inventory_problem_with_tuples(self):
        """
        Inventory Problem with tuple arcs and string-indexed sets.
        """
        model_code = """
        // Inventory Problem with tuple arcs and string-indexed sets.

        range Periods = 1..3;

        tuple Store {
        string name;
        }

        {Store} Stores;

        int Capacity[Stores];
        int Demand[Stores][Periods];
        int TransportCost[Stores][Periods];
        int HoldingCost;

        dvar int Inventory[Stores][Periods];
        dvar int Shipments[Stores][Periods];

        minimize sum(s in Stores, p in Periods) (TransportCost[s][p] * Shipments[s][p] + HoldingCost * Inventory[s][p]);

        subject to {
        forall(s in Stores)
            Inventory[s][1] == 0 + Shipments[s][1] - Demand[s][1];
        forall(s in Stores, p in 2..3)
            Inventory[s][p] == Inventory[s][p-1] + Shipments[s][p] - Demand[s][p];

        forall(s in Stores, p in Periods) {
            Inventory[s][p] <= Capacity[s];
        }

        forall(s in Stores, p in Periods) {
            Inventory[s][p] >= 0;
        }

        forall(s in Stores, p in Periods)
            Shipments[s][p] >= 0;
        }


        """
        data_code = """
        // Data for the inventory problem

        Stores = { <"StoreA">, <"StoreB"> };

        // Capacity per store (keys must match Stores tuple elements)
        Capacity = [
            <"StoreA"> 100,
            <"StoreB"> 100
        ];

        // Demand per store and period, provided as a 2D array (rows aligned with Stores order)
        Demand = [
            [1, 2, 3],
            [4, 5, 6]
        ];

        // TransportCost per store and period, provided as a 2D array (rows aligned with Stores order)
        TransportCost = [
            [10, 12, 15],
            [8, 11, 13]
        ];

        HoldingCost = 1;
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_shortest_path_with_tuples(self):
        """
        Shortest Path with tuple arcs.
        """
        model_code = """
        // Shortest Path with tuple arcs.

        tuple Arc { int from; int to; float cost; }

        int N = ...;
        range Nodes = 1..N;
        {Arc} arcs = ...;
        int source = ...;
        int dest = ...;
        dvar int+ x[arcs];

        minimize sum(a in arcs) a.cost * x[a];

        subject to {
            forall(i in Nodes) (
                sum(a in arcs: a.from == i) x[a] - sum(a in arcs: a.to == i) x[a] == ((i == source) ? 1 : ((i == dest) ? -1 : 0))
            );
        }
        """
        data_code = """
        // Data for the inshortest pathventory problem

        N = 5;
        source = 1;
        dest = 5;
        arcs = {
        <1, 2, 2.0>,
        <1, 3, 3.0>,
        <2, 3, 1.0>,
        <2, 4, 1.0>,
        <3, 4, 1.0>,
        <4, 5, 2.0>,
        <3, 5, 5.0>
        };
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_shortest_path_with_tuples_and_strings(self):
        """
        Shortest Path with tuple arcs and strings.
        """
        model_code = """
        // Shortest Path with tuple arcs and strings.

        tuple Arc { string from; string to; float cost; }

        {string} Cities = ...;

        {Arc} arcs = ...;
        string source = ...;
        string dest = ...;
        dvar int+ x[arcs];

        minimize sum(a in arcs) a.cost * x[a];

        subject to {
        forall(i in Cities) (
            sum(a in arcs: a.from == i) x[a] - sum(a in arcs: a.to == i) x[a] == ((i == source) ? 1 : ((i == dest) ? -1 : 0))
        );
        }
        """
        data_code = """
        // Data for the shortest path problem

        Cities = { "London", "Oxford", "Cambridge",
           "Norwich", "Birmingham", "Manchester" };
        source = "London";
        dest = "Birmingham";
        arcs = {
        <"London", "Oxford", 90.0>,
        <"London", "Cambridge", 100.0>,
        <"London", "Norwich", 180.0>,
        <"London", "Birmingham", 205.0>,
        <"London", "Manchester", 335.0>,
        <"Oxford", "London", 90.0>,
        <"Oxford", "Cambridge", 140.0>,
        <"Oxford", "Norwich", 220.0>,
        <"Oxford", "Birmingham", 110.0>,
        <"Oxford", "Manchester", 260.0>,
        <"Cambridge", "London", 100.0>,
        <"Cambridge", "Oxford", 140.0>,
        <"Cambridge", "Norwich", 100.0>,
        <"Cambridge", "Birmingham", 160.0>,
        <"Cambridge", "Manchester", 250.0>,
        <"Norwich", "London", 180.0>,
        <"Norwich", "Oxford", 220.0>,
        <"Norwich", "Cambridge", 100.0>,
        <"Norwich", "Birmingham", 240.0>,
        <"Norwich", "Manchester", 350.0>,
        <"Birmingham", "London", 205.0>,
        <"Birmingham", "Oxford", 110.0>,
        <"Birmingham", "Cambridge", 160.0>,
        <"Birmingham", "Norwich", 240.0>,
        <"Birmingham", "Manchester", 140.0>,
        <"Manchester", "London", 335.0>,
        <"Manchester", "Oxford", 260.0>,
        <"Manchester", "Cambridge", 250.0>,
        <"Manchester", "Norwich", 350.0>,
        <"Manchester", "Birmingham", 140.0>
        };
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)

    def test_logistics_with_tuples_and_strings(self):
        """
        Logistics with tuple arcs and strings.
        """
        model_code = """
        // Logistics with tuple arcs and strings.

        {string} Factories = { "F1", "F2", "F3", "F4" };
        {string} Warehouses = { "W1", "W2", "W3", "W4" };

        float cost[Factories][Warehouses];
        int supply[Factories];
        int demand[Warehouses];

        dvar int+ x[Factories][Warehouses];

        minimize sum(f in Factories, w in Warehouses) cost[f][w] * x[f][w];

        subject to {
            forall(f in Factories)
                sum(w in Warehouses) x[f][w] == supply[f];

            forall(w in Warehouses)
                sum(f in Factories) x[f][w] == demand[w];

            // Restrict F1 to only supply W1 and W2
            forall(f in Factories, w in Warehouses : f == "F1" && w != "W1" && w != "W2")
                x[f][w] == 0;

            // Shut down F1
            forall(w in Warehouses) x["F1"][w] <= 30;
        }
        """
        data_code = """
        // Data for the logistics problem

        cost = [ [0, 100, 400, 200],
                [100, 0, 300, 400],
                [400, 300, 0, 700],
                [200, 400, 700, 0] ];
        supply = [50, 20, 90, 30];
        demand = [40, 40, 60, 50];
        """
        import os
        import tempfile

        from pyopl.pyopl_core import solve

        obj_values = {}
        for solver in ("gurobi", "scipy"):
            with (
                tempfile.NamedTemporaryFile("w", suffix=".mod", delete=False) as tmp_mod,
                tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False) as tmp_dat,
            ):
                tmp_mod.write(model_code)
                tmp_mod.flush()
                model_file = tmp_mod.name
                tmp_dat.write(data_code)
                tmp_dat.flush()
                data_file = tmp_dat.name
            try:
                result = solve(model_file, data_file, solver=solver)
                self.assertNotEqual(result.get("status"), "FAILED", f"Solver {solver} failed: {result}")
                self.assertIn("objective_value", result)
                obj = result["objective_value"]
                obj_values[solver] = obj
            finally:
                if os.path.exists(model_file):
                    os.remove(model_file)
                if os.path.exists(data_file):
                    os.remove(data_file)
        if "gurobi" in obj_values and "scipy" in obj_values:
            self.assertAlmostEqual(obj_values["gurobi"], obj_values["scipy"], places=6)
