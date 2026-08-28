# Commands — Full exhaustive reference (verbatim structs)

Source: `rts/ExternalAI/Interface/AISCommands.h` (verbatim excerpt)

This file contains the full command structs and comments exactly as defined in the engine interface. Use this as the authoritative reference when building command data for `handleCommand()` / `GiveOrder()` or to map AI-level commands to engine internals.

---

```cpp
/* (excerpt—full verbatim structs from rts/ExternalAI/Interface/AISCommands.h) */

struct SSetMyIncomeMultiplierCheatCommand {
	/// default: 1.0; common: [0.0, 2.0]; valid: [0.0, FLOAT_MAX]
	float factor;
}; //$ COMMAND_CHEATS_SET_MY_INCOME_MULTIPLIER Cheats_setMyIncomeMultiplier

struct SGiveMeResourceCheatCommand {
	int resourceId;
	float amount;
}; //$ COMMAND_CHEATS_GIVE_ME_RESOURCE Cheats_giveMeResource REF:resourceId->Resource

struct SGiveMeNewUnitCheatCommand {
	int unitDefId;
	float* pos_posF3;
	int ret_newUnitId;
}; //$ COMMAND_CHEATS_GIVE_ME_NEW_UNIT Cheats_giveMeUnit REF:unitDefId->UnitDef REF:ret_newUnitId->Unit

struct SSendTextMessageCommand {
	const char* text;
	int zone;
}; //$ COMMAND_SEND_TEXT_MESSAGE Game_sendTextMessage

struct SSetLastPosMessageCommand {
	float* pos_posF3;
}; //$ COMMAND_SET_LAST_POS_MESSAGE Game_setLastMessagePosition

struct SSendResourcesCommand {
	int resourceId;
	float amount;
	int receivingTeamId;
	bool ret_isExecuted;
}; //$ COMMAND_SEND_RESOURCES Economy_sendResource REF:resourceId->Resource REF:receivingTeamId->Team

struct SSendUnitsCommand {
	int* unitIds;
	int unitIds_size;
	int receivingTeamId;
	int ret_sentUnits;
}; //$ COMMAND_SEND_UNITS Economy_sendUnits REF:MULTI:unitIds->Unit REF:receivingTeamId->Team

struct SCreateGroupCommand {
	int ret_groupId;
}; //$ COMMAND_GROUP_CREATE Group_create REF:ret_groupId->Group STATIC

struct SEraseGroupCommand {
	int groupId;
}; //$ COMMAND_GROUP_ERASE Group_erase REF:groupId->Group

struct SInitPathCommand {
	float* start_posF3;
	float* end_posF3;
	int pathType;
	float goalRadius;
	int ret_pathId;
}; //$ COMMAND_PATH_INIT Pathing_initPath REF:ret_pathId->Path

struct SGetApproximateLengthPathCommand {
	float* start_posF3;
	float* end_posF3;
	int pathType;
	float goalRadius;
	float ret_approximatePathLength;
}; //$ COMMAND_PATH_GET_APPROXIMATE_LENGTH Pathing_getApproximateLength

struct SGetNextWaypointPathCommand {
	int pathId;
	float* ret_nextWaypoint_posF3_out;
}; //$ COMMAND_PATH_GET_NEXT_WAYPOINT Pathing_getNextWaypoint REF:pathId->Path

struct SFreePathCommand {
	int pathId;
}; //$ COMMAND_PATH_FREE Pathing_freePath REF:pathId->Path

struct SCallLuaRulesCommand {
	const char* inData;
	int inSize;
	char* ret_outData;
}; //$ COMMAND_CALL_LUA_RULES Lua_callRules

struct SCallLuaUICommand {
	const char* inData;
	int inSize;
	char* ret_outData;
}; //$ COMMAND_CALL_LUA_UI Lua_callUI

struct SSendStartPosCommand {
	bool ready;
	float* pos_posF3;
}; //$ COMMAND_SEND_START_POS Game_sendStartPosition

struct SAddNotificationDrawerCommand {
	float* pos_posF3;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_ADD_NOTIFICATION Map_Drawer_addNotification

struct SAddPointDrawCommand {
	float* pos_posF3;
	const char* label;
}; //$ COMMAND_DRAWER_POINT_ADD Map_Drawer_addPoint

struct SRemovePointDrawCommand {
	float* pos_posF3;
}; //$ COMMAND_DRAWER_POINT_REMOVE Map_Drawer_deletePointsAndLines

struct SAddLineDrawCommand {
	float* posFrom_posF3;
	float* posTo_posF3;
}; //$ COMMAND_DRAWER_LINE_ADD Map_Drawer_addLine

struct SStartPathDrawerCommand {
	float* pos_posF3;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_PATH_START Map_Drawer_PathDrawer_start

struct SFinishPathDrawerCommand {
	bool iAmUseless;
}; //$ COMMAND_DRAWER_PATH_FINISH Map_Drawer_PathDrawer_finish

struct SDrawLinePathDrawerCommand {
	float* endPos_posF3;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_PATH_DRAW_LINE Map_Drawer_PathDrawer_drawLine

struct SDrawLineAndIconPathDrawerCommand {
	int cmdId;
	float* endPos_posF3;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_PATH_DRAW_LINE_AND_ICON Map_Drawer_PathDrawer_drawLineAndCommandIcon REF:cmdId->Command

struct SDrawIconAtLastPosPathDrawerCommand {
	int cmdId;
}; //$ COMMAND_DRAWER_PATH_DRAW_ICON_AT_LAST_POS Map_Drawer_PathDrawer_drawIcon REF:cmdId->Command

struct SBreakPathDrawerCommand {
	float* endPos_posF3;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_PATH_BREAK Map_Drawer_PathDrawer_suspend

struct SRestartPathDrawerCommand {
	bool sameColor;
}; //$ COMMAND_DRAWER_PATH_RESTART Map_Drawer_PathDrawer_restart

struct SCreateSplineFigureDrawerCommand {
	float* pos1_posF3;
	float* pos2_posF3;
	float* pos3_posF3;
	float* pos4_posF3;
	float width;
	bool arrow;
	int lifeTime;
	int figureGroupId;
	int ret_newFigureGroupId;
}; //$ COMMAND_DRAWER_FIGURE_CREATE_SPLINE Map_Drawer_Figure_drawSpline REF:figureGroupId->FigureGroup REF:ret_newFigureGroupId->FigureGroup

struct SCreateLineFigureDrawerCommand {
	float* pos1_posF3;
	float* pos2_posF3;
	float width;
	bool arrow;
	int lifeTime;
	int figureGroupId;
	int ret_newFigureGroupId;
}; //$ COMMAND_DRAWER_FIGURE_CREATE_LINE Map_Drawer_Figure_drawLine REF:figureGroupId->FigureGroup REF:ret_newFigureGroupId->FigureGroup

struct SSetColorFigureDrawerCommand {
	int figureGroupId;
	short* color_colorS3;
	short alpha;
}; //$ COMMAND_DRAWER_FIGURE_SET_COLOR Map_Drawer_Figure_setColor REF:figureGroupId->FigureGroup

struct SDeleteFigureDrawerCommand {
	int figureGroupId;
}; //$ COMMAND_DRAWER_FIGURE_DELETE Map_Drawer_Figure_remove REF:figureGroupId->FigureGroup

struct SDrawUnitDrawerCommand {
	int toDrawUnitDefId;
	float* pos_posF3;
	float rotation;
	int lifeTime;
	int teamId;
	bool transparent;
	bool drawBorder;
	int facing;
}; //$ COMMAND_DRAWER_DRAW_UNIT Map_Drawer_drawUnit REF:toDrawUnitDefId->UnitDef

struct SBuildUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int toBuildUnitDefId;
	float* buildPos_posF3;
	int facing;
}; //$ COMMAND_UNIT_BUILD Unit_build REF:toBuildUnitDefId->UnitDef

struct SStopUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
}; //$ COMMAND_UNIT_STOP Unit_stop

struct SWaitUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
}; //$ COMMAND_UNIT_WAIT Unit_wait

struct STimeWaitUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
	int time;
}; //$ COMMAND_UNIT_WAIT_TIME Unit_waitFor

struct SDeathWaitUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
	int toDieUnitId;
}; //$ COMMAND_UNIT_WAIT_DEATH Unit_waitForDeathOf REF:toDieUnitId->Unit

struct SSquadWaitUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int numUnits;
}; //$ COMMAND_UNIT_WAIT_SQUAD Unit_waitForSquadSize

struct SGatherWaitUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
}; //$ COMMAND_UNIT_WAIT_GATHER Unit_waitForAll

struct SMoveUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	float* toPos_posF3;
}; //$ COMMAND_UNIT_MOVE Unit_moveTo

struct SPatrolUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	float* toPos_posF3;
}; //$ COMMAND_UNIT_PATROL Unit_patrolTo

struct SFightUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	float* toPos_posF3;
}; //$ COMMAND_UNIT_FIGHT Unit_fight

struct SAttackUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int toAttackUnitId;
}; //$ COMMAND_UNIT_ATTACK Unit_attack REF:toAttackUnitId->Unit

struct SAttackAreaUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	float* toAttackPos_posF3;
	float radius;
}; //$ COMMAND_UNIT_ATTACK_AREA Unit_attackArea

struct SGuardUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int toGuardUnitId;
}; //$ COMMAND_UNIT_GUARD Unit_guard REF:toGuardUnitId->Unit

struct SGroupAddUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int toGroupId;
}; //$ COMMAND_UNIT_GROUP_ADD Unit_addToGroup REF:toGroupId->Group

struct SGroupClearUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
}; //$ COMMAND_UNIT_GROUP_CLEAR Unit_removeFromGroup

struct SRepairUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int toRepairUnitId;
}; //$ COMMAND_UNIT_REPAIR Unit_repair REF:toRepairUnitId->Unit

struct SSetFireStateUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int fireState;
}; //$ COMMAND_UNIT_SET_FIRE_STATE Unit_setFireState

struct SSetMoveStateUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int moveState;
}; //$ COMMAND_UNIT_SET_MOVE_STATE Unit_setMoveState

struct SSetBaseUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	float* basePos_posF3;
}; //$ COMMAND_UNIT_SET_BASE Unit_setBase

struct SSelfDestroyUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;
}; //$ COMMAND_UNIT_SELF_DESTROY Unit_selfDestruct

struct SLoadUnitsUnitCommand {
	int unitId;
	int groupId;
	short options;
	int timeOut;

	int* toLoadUnitIds;
	int toLoadUnitIds_size;
}; //$ COMMAND_UNIT_LOAD_UNITS Unit_loadUnits REF:MULTI:toLoadUnitIds->Unit

struct SLoad]]