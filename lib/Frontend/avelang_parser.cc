#include "avelang_parser.h"
#include "avelang_parser_utils.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include <clang/Basic/FileSystemOptions.h>
#include <llvm/Support/MemoryBuffer.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace causalflow::avelang::frontend {

using ASTNode = ast::ASTNode;

AveLangParser::AveLangParser(
    llvm::IntrusiveRefCntPtr<ast::ASTContext> context,
    llvm::IntrusiveRefCntPtr<basic::DiagnosticManager> diagnostic_manager)
    : context_(context), diagnostic_manager_(diagnostic_manager),
      root_node_(nullptr) {

    PythonInitializer::GetInstance().EnsureInitialized();
}

void AveLangParser::Parse(const std::string &file_name) {
    auto &file_manager = diagnostic_manager_->GetFileManager();
    try {
        auto file_ref = file_manager.getFileRef(file_name, true);
        if (!file_ref) {
            diagnostic_manager_->GetEngine()->Report(
                clang::diag::err_cannot_open_file)
                << file_name;
            return;
        }

        // Get the buffer from the file
        auto buffer_or_error = file_manager.getBufferForFile(file_ref.get());
        if (!buffer_or_error) {
            diagnostic_manager_->GetEngine()->Report(
                clang::diag::err_cannot_open_file)
                << file_name;
            return;
        }

        // Use ParseFromBuffer with the file's buffer
        ParseFromBuffer(**buffer_or_error, file_name);

    } catch (const std::exception &e) {
        diagnostic_manager_->GetEngine()->Report(
            clang::diag::err_cannot_open_file)
            << file_name << e.what();
    }
}

void AveLangParser::ParseFromBuffer(const llvm::MemoryBuffer &buffer,
                                    llvm::StringRef buffer_name) {
    auto &source_manager = diagnostic_manager_->GetSourceManager();
    try {
        // Create main file ID for source location tracking from MemoryBuffer
        clang::FileID main_file_id = source_manager.createFileID(
            llvm::MemoryBufferRef(buffer), clang::SrcMgr::C_User);
        source_manager.setMainFileID(main_file_id);

        llvm::StringRef source_code = buffer.getBuffer();

        ParseImpl(main_file_id, buffer_name.str(), source_code);

    } catch (const std::exception &e) {
        diagnostic_manager_->GetEngine()->Report(
            clang::diag::err_cannot_open_file)
            << buffer_name << e.what();
    }
}

static clang::FileID
PrepareSourceManager(basic::DiagnosticManager *diagnostic_manager,
                     llvm::StringRef file_name, llvm::StringRef source_code,
                     std::unique_ptr<llvm::MemoryBuffer> &owned_buffer) {
    auto &source_manager = diagnostic_manager->GetSourceManager();
    // SourceManager keeps only a MemoryBufferRef, so the backing bytes must
    // outlive diagnostics rendering. Copy Python-provided source text into an
    // owned buffer instead of referencing a temporary std::string.
    owned_buffer = llvm::MemoryBuffer::getMemBufferCopy(source_code, file_name);
    clang::FileID file_id = source_manager.createFileID(
        llvm::MemoryBufferRef(*owned_buffer), clang::SrcMgr::C_User);
    source_manager.setMainFileID(file_id);
    return file_id;
}

static std::pair<std::string, std::string>
GetPythonSourceMetadata(const py::object &py_ast_node,
                        llvm::StringRef default_file_name) {
    std::string file_name = default_file_name.str();
    std::string source_code;

    if (py::hasattr(py_ast_node, "_avelang_file_name")) {
        py::object value = py_ast_node.attr("_avelang_file_name");
        if (!value.is_none()) {
            file_name = value.cast<std::string>();
        }
    }

    if (py::hasattr(py_ast_node, "_avelang_source_code")) {
        py::object value = py_ast_node.attr("_avelang_source_code");
        if (!value.is_none()) {
            source_code = value.cast<std::string>();
        }
    }

    return {std::move(file_name), std::move(source_code)};
}

ast::ASTNode *AveLangParser::ParsePythonAST(const py::object &py_ast_node,
                                            llvm::StringRef file_name) {
    auto [resolved_file_name, source_code] =
        GetPythonSourceMetadata(py_ast_node, file_name);
    current_file_id_ =
        PrepareSourceManager(diagnostic_manager_.get(), resolved_file_name,
                             source_code, owned_buffer_);
    root_node_ = ParseRoot(py_ast_node);
    return root_node_;
}

ast::FunctionDef *
AveLangParser::ParsePythonFunctionDef(const py::object &py_function_def,
                                      llvm::StringRef file_name) {
    auto [resolved_file_name, source_code] =
        GetPythonSourceMetadata(py_function_def, file_name);
    current_file_id_ =
        PrepareSourceManager(diagnostic_manager_.get(), resolved_file_name,
                             source_code, owned_buffer_);
    clang::SourceRange range = ExtractSourceRange(py_function_def);
    ParsingContextGuard context_guard(this, range);
    ast::FunctionDef *func = ParseFunctionDef(py_function_def);
    if (func) {
        func->SetSourceRange(range);
    }
    return func;
}

std::unique_ptr<llvm::MemoryBuffer> AveLangParser::ReleaseOwnedBuffer() {
    return std::move(owned_buffer_);
}

void AveLangParser::ParseImpl(const clang::FileID main_file_id,
                              const std::string &file_name,
                              const llvm::StringRef &source_code) {
    current_file_id_ = main_file_id;

    try {
        py::module_ ast = py::module_::import("ast");
        py::object parsed_ast = ast.attr("parse")(source_code.str(), file_name);

        // Parse the Python AST using the dispatcher
        root_node_ = ParseRoot(parsed_ast);

    } catch (const py::error_already_set &e) {
        diagnostic_manager_->GetEngine()->Report(
            clang::diag::err_cannot_open_file)
            << file_name << e.what();
    }
}

ASTNode *AveLangParser::ParseRoot(const py::object &py_ast_node) {
    if (!py_ast_node || py_ast_node.is_none()) {
        return nullptr;
    }

    clang::SourceRange range = ExtractSourceRange(py_ast_node);

    // Use RAII guard to automatically manage source range stack
    ParsingContextGuard context_guard(this, range);

    static const std::unordered_map<
        std::string,
        std::function<ASTNode *(AveLangParser *, const py::object &)>>
        kPyASTClassParser = {
            {"Module", &AveLangParser::ParseModule},
        };

    // Get the class name of the Python AST node
    py::object py_class = py_ast_node.attr("__class__");
    std::string class_name = py_class.attr("__name__").cast<std::string>();

    // Dispatch to appropriate parsing function based on class name
    auto it = kPyASTClassParser.find(class_name);
    ASTNode *ast_node = nullptr;
    if (it != kPyASTClassParser.end()) {
        ast_node = it->second(this, py_ast_node);
    } else {
        Report(basic::DiagnosticCode::kUnknownASTNode) << class_name;
    }

    if (ast_node) {
        ast_node->SetSourceRange(range);
    }

    return ast_node;
}

ast::Module *AveLangParser::ParseModule(const py::object &py_module) {
    // Parse module body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_module, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Create Module with constructor
    ast::Module *module = context_->Allocate<ast::Module>();
    new (module) ast::Module(std::move(body));
    return module;
}

ast::FunctionDef *
AveLangParser::ParseFunctionDef(const py::object &py_function_def) {
    // Extract function name
    std::string name;
    if (py::hasattr(py_function_def, "name")) {
        name = py_function_def.attr("name").cast<std::string>();
    }

    // Parse decorator list
    llvm::SmallVector<ast::Expr *, 4> decorators;
    for (auto decorator : MaybePyList(py_function_def, "decorator_list")) {
        ast::Expr *parsed_decorator = ParseExpr(decorator.cast<py::object>());
        if (parsed_decorator) {
            decorators.push_back(parsed_decorator);
        }
    }

    // Parse arguments - py_function_def.args is an arguments object
    ast::Arguments *args = nullptr;
    if (py::hasattr(py_function_def, "args")) {
        py::object args_obj = py_function_def.attr("args");
        args = ParseArguments(args_obj);
    }

    // Parse return type annotation if present
    ast::Expr *returns = nullptr;
    if (py::hasattr(py_function_def, "returns")) {
        py::object returns_obj = py_function_def.attr("returns");
        if (!returns_obj.is_none()) {
            returns = ParseExpr(returns_obj);
        }
    }

    // Parse function body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_function_def, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Create FunctionDef with constructor
    ast::FunctionDef *function_def = context_->Allocate<ast::FunctionDef>();
    new (function_def) ast::FunctionDef(std::move(name), args, std::move(body),
                                        std::move(decorators), returns);
    return function_def;
}

ast::Arguments *AveLangParser::ParseArguments(const py::object &py_arguments) {
    // Collect all argument components
    llvm::SmallVector<ast::Arg *, 4> posonlyargs;
    llvm::SmallVector<ast::Arg *, 4> args;
    llvm::SmallVector<ast::Arg *, 4> kwonlyargs;
    llvm::SmallVector<ast::Expr *, 4> defaults;
    llvm::SmallVector<ast::Expr *, 4> kw_defaults;
    ast::Arg *vararg = nullptr;
    ast::Arg *kwarg = nullptr;

    // Parse positional-only arguments (posonlyargs)
    for (auto arg : MaybePyList(py_arguments, "posonlyargs")) {
        ast::Arg *parsed_arg = ParseArg(arg.cast<py::object>());
        if (parsed_arg) {
            posonlyargs.push_back(parsed_arg);
        }
    }

    // Parse regular arguments (args)
    for (auto arg : MaybePyList(py_arguments, "args")) {
        ast::Arg *parsed_arg = ParseArg(arg.cast<py::object>());
        if (parsed_arg) {
            args.push_back(parsed_arg);
        }
    }

    // Parse keyword-only arguments (kwonlyargs)
    for (auto arg : MaybePyList(py_arguments, "kwonlyargs")) {
        ast::Arg *parsed_arg = ParseArg(arg.cast<py::object>());
        if (parsed_arg) {
            kwonlyargs.push_back(parsed_arg);
        }
    }

    // Parse default values (defaults)
    for (auto default_val : MaybePyList(py_arguments, "defaults")) {
        ast::Expr *parsed_default = ParseExpr(default_val.cast<py::object>());
        if (parsed_default) {
            defaults.push_back(parsed_default);
        }
    }

    // Parse keyword-only defaults (kw_defaults)
    for (auto kw_default : MaybePyList(py_arguments, "kw_defaults")) {
        ast::Expr *parsed_kw_default = ParseExpr(kw_default.cast<py::object>());
        if (parsed_kw_default) {
            kw_defaults.push_back(parsed_kw_default);
        }
    }

    // Parse vararg (*args)
    if (py::hasattr(py_arguments, "vararg") &&
        !py_arguments.attr("vararg").is_none()) {
        py::object vararg_obj = py_arguments.attr("vararg");
        vararg = ParseArg(vararg_obj);
    }

    // Parse kwarg (**kwargs)
    if (py::hasattr(py_arguments, "kwarg") &&
        !py_arguments.attr("kwarg").is_none()) {
        py::object kwarg_obj = py_arguments.attr("kwarg");
        kwarg = ParseArg(kwarg_obj);
    }

    // Create and return Arguments with constructor
    ast::Arguments *arguments = context_->Allocate<ast::Arguments>();
    new (arguments) ast::Arguments(std::move(posonlyargs), std::move(args),
                                   std::move(kwonlyargs), std::move(defaults),
                                   std::move(kw_defaults), vararg, kwarg);
    return arguments;
}

ast::Return *AveLangParser::ParseReturn(const py::object &py_return) {
    // Parse return value if present
    ast::Expr *value = nullptr;
    if (py::hasattr(py_return, "value") && !py_return.attr("value").is_none()) {
        py::object value_obj = py_return.attr("value");
        value = ParseExpr(value_obj);
    }

    // Create Return with constructor
    ast::Return *return_stmt = context_->Allocate<ast::Return>();
    new (return_stmt) ast::Return(value);
    return return_stmt;
}

ast::Stmt *AveLangParser::ParseStmt(const py::object &py_stmt) {
    if (!py_stmt || py_stmt.is_none()) {
        return nullptr;
    }

    clang::SourceRange range = ExtractSourceRange(py_stmt);

    // Use RAII guard to automatically manage source range stack
    ParsingContextGuard context_guard(this, range);

    // Statement dispatch map
    static const std::unordered_map<
        std::string,
        std::function<ast::Stmt *(AveLangParser *, const py::object &)>>
        kStmtParser = {
            {"FunctionDef", &AveLangParser::ParseFunctionDef},
            {"Return", &AveLangParser::ParseReturn},
            {"Assign", &AveLangParser::ParseAssign},
            {"AugAssign", &AveLangParser::ParseAugAssign},
            {"AnnAssign", &AveLangParser::ParseAnnAssign},
            {"For", &AveLangParser::ParseFor},
            {"While", &AveLangParser::ParseWhile},
            {"If", &AveLangParser::ParseIf},
            {"With", &AveLangParser::ParseWith},
            {"Pass", &AveLangParser::ParsePass},
            {"Break", &AveLangParser::ParseBreak},
            {"Continue", &AveLangParser::ParseContinue},
            {"Expr", &AveLangParser::ParseExprStmt},
            {"Import", &AveLangParser::ParseImport},
            {"ImportFrom", &AveLangParser::ParseImportFrom},
        };

    // Get the class name of the Python AST node
    py::object py_class = py_stmt.attr("__class__");
    std::string class_name = py_class.attr("__name__").cast<std::string>();

    // Dispatch to appropriate parsing function based on class name
    auto it = kStmtParser.find(class_name);
    ast::Stmt *stmt = nullptr;
    if (it != kStmtParser.end()) {
        stmt = it->second(this, py_stmt);
    }

    if (stmt) {
        stmt->SetSourceRange(range);
    }

    return stmt;
}

clang::SourceRange
AveLangParser::ExtractSourceRange(const py::object &py_ast_node) {
    clang::SourceLocation start_loc, end_loc;
    auto &source_manager = diagnostic_manager_->GetSourceManager();

    // Extract start location from Python AST
    if (py::hasattr(py_ast_node, "lineno") &&
        py::hasattr(py_ast_node, "col_offset")) {
        int line_no = py_ast_node.attr("lineno").cast<int>();
        int col_offset = py_ast_node.attr("col_offset").cast<int>();
        start_loc = source_manager.translateLineCol(current_file_id_, line_no,
                                                    col_offset + 1);
    } else {
        start_loc = source_manager.getLocForStartOfFile(current_file_id_);
    }

    // Extract end location from Python AST
    if (py::hasattr(py_ast_node, "end_lineno") &&
        py::hasattr(py_ast_node, "end_col_offset")) {
        int end_line_no = py_ast_node.attr("end_lineno").cast<int>();
        int end_col_offset = py_ast_node.attr("end_col_offset").cast<int>();
        end_loc = source_manager.translateLineCol(current_file_id_, end_line_no,
                                                  end_col_offset + 1);
    } else {
        end_loc = source_manager.getLocForEndOfFile(current_file_id_);
    }

    return clang::SourceRange(start_loc, end_loc);
}

clang::SourceRange AveLangParser::GetCurrentSourceRange() const {
    if (!source_range_stack_.empty()) {
        return source_range_stack_.back();
    }
    return clang::SourceRange();
}

clang::DiagnosticBuilder
AveLangParser::Report(basic::DiagnosticCode error_code) {
    clang::SourceLocation loc;
    if (!source_range_stack_.empty()) {
        loc = source_range_stack_.back().getBegin();
    }
    return diagnostic_manager_->Report(error_code, loc);
}

ast::Assign *AveLangParser::ParseAssign(const py::object &py_assign) {
    // Parse targets (left-hand side of assignment)
    llvm::SmallVector<ast::Expr *, 4> targets;
    for (auto target : MaybePyList(py_assign, "targets")) {
        ast::Expr *parsed_target = ParseExpr(target.cast<py::object>());
        if (parsed_target) {
            targets.push_back(parsed_target);
        }
    }

    // Parse value (right-hand side of assignment)
    ast::Expr *value = nullptr;
    if (py::hasattr(py_assign, "value")) {
        py::object value_obj = py_assign.attr("value");
        value = ParseExpr(value_obj);
    }

    // Create Assign with constructor
    ast::Assign *assign = context_->Allocate<ast::Assign>();
    new (assign) ast::Assign(std::move(targets), value);
    return assign;
}

ast::AugAssign *AveLangParser::ParseAugAssign(const py::object &py_augassign) {
    // Parse target
    ast::Expr *target = nullptr;
    if (py::hasattr(py_augassign, "target")) {
        py::object target_obj = py_augassign.attr("target");
        target = ParseExpr(target_obj);
    }

    // Parse operator
    std::string op;
    if (py::hasattr(py_augassign, "op")) {
        py::object op_obj = py_augassign.attr("op");
        py::object op_class = op_obj.attr("__class__");
        op = op_class.attr("__name__").cast<std::string>();
    }

    // Parse value
    ast::Expr *value = nullptr;
    if (py::hasattr(py_augassign, "value")) {
        py::object value_obj = py_augassign.attr("value");
        value = ParseExpr(value_obj);
    }

    // Create AugAssign with constructor
    ast::AugAssign *augassign = context_->Allocate<ast::AugAssign>();
    new (augassign) ast::AugAssign(target, std::move(op), value);
    return augassign;
}

ast::AnnAssign *AveLangParser::ParseAnnAssign(const py::object &py_annassign) {
    // Parse target
    ast::Expr *target = nullptr;
    if (py::hasattr(py_annassign, "target")) {
        py::object target_obj = py_annassign.attr("target");
        target = ParseExpr(target_obj);
    }

    // Parse annotation
    ast::Expr *annotation = nullptr;
    if (py::hasattr(py_annassign, "annotation")) {
        py::object annotation_obj = py_annassign.attr("annotation");
        annotation = ParseExpr(annotation_obj);
    }

    // Parse optional value
    ast::Expr *value = nullptr;
    if (py::hasattr(py_annassign, "value") &&
        !py_annassign.attr("value").is_none()) {
        py::object value_obj = py_annassign.attr("value");
        value = ParseExpr(value_obj);
    }

    // Parse simple flag
    bool simple = true;
    if (py::hasattr(py_annassign, "simple")) {
        simple = py_annassign.attr("simple").cast<bool>();
    }

    // Create AnnAssign with constructor
    ast::AnnAssign *annassign = context_->Allocate<ast::AnnAssign>();
    new (annassign) ast::AnnAssign(target, annotation, value, simple);
    return annassign;
}

ast::For *AveLangParser::ParseFor(const py::object &py_for) {
    // Parse target
    ast::Expr *target = nullptr;
    if (py::hasattr(py_for, "target")) {
        py::object target_obj = py_for.attr("target");
        target = ParseExpr(target_obj);
    }

    // Parse iter
    ast::Expr *iter = nullptr;
    if (py::hasattr(py_for, "iter")) {
        py::object iter_obj = py_for.attr("iter");
        iter = ParseExpr(iter_obj);
    }

    // Parse body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_for, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Parse orelse
    llvm::SmallVector<ast::Stmt *, 4> orelse;
    for (auto stmt : MaybePyList(py_for, "orelse")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            orelse.push_back(parsed_stmt);
        }
    }

    // Create For with constructor
    ast::For *for_stmt = context_->Allocate<ast::For>();
    new (for_stmt) ast::For(target, iter, std::move(body), std::move(orelse));
    return for_stmt;
}

ast::While *AveLangParser::ParseWhile(const py::object &py_while) {
    // Parse test
    ast::Expr *test = nullptr;
    if (py::hasattr(py_while, "test")) {
        py::object test_obj = py_while.attr("test");
        test = ParseExpr(test_obj);
    }

    // Parse body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_while, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Parse orelse
    llvm::SmallVector<ast::Stmt *, 4> orelse;
    for (auto stmt : MaybePyList(py_while, "orelse")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            orelse.push_back(parsed_stmt);
        }
    }

    // Create While with constructor
    ast::While *while_stmt = context_->Allocate<ast::While>();
    new (while_stmt) ast::While(test, std::move(body), std::move(orelse));
    return while_stmt;
}

ast::If *AveLangParser::ParseIf(const py::object &py_if) {
    // Parse test
    ast::Expr *test = nullptr;
    if (py::hasattr(py_if, "test")) {
        py::object test_obj = py_if.attr("test");
        test = ParseExpr(test_obj);
    }

    // Parse body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_if, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Parse orelse
    llvm::SmallVector<ast::Stmt *, 4> orelse;
    for (auto stmt : MaybePyList(py_if, "orelse")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            orelse.push_back(parsed_stmt);
        }
    }

    // Create If with constructor
    ast::If *if_stmt = context_->Allocate<ast::If>();
    new (if_stmt) ast::If(test, std::move(body), std::move(orelse));
    return if_stmt;
}

ast::With *AveLangParser::ParseWith(const py::object &py_with) {
    // Parse items (with context managers)
    llvm::SmallVector<ast::With::WithItem, 4> items;
    for (auto item : MaybePyList(py_with, "items")) {
        py::object item_obj = item.cast<py::object>();

        // Parse context_expr
        ast::Expr *context_expr = nullptr;
        if (py::hasattr(item_obj, "context_expr")) {
            py::object context_obj = item_obj.attr("context_expr");
            context_expr = ParseExpr(context_obj);
        }

        // Parse optional_vars
        ast::Expr *optional_vars = nullptr;
        if (py::hasattr(item_obj, "optional_vars") &&
            !item_obj.attr("optional_vars").is_none()) {
            py::object vars_obj = item_obj.attr("optional_vars");
            optional_vars = ParseExpr(vars_obj);
        }

        items.emplace_back(context_expr, optional_vars);
    }

    // Parse body
    llvm::SmallVector<ast::Stmt *, 4> body;
    for (auto stmt : MaybePyList(py_with, "body")) {
        ast::Stmt *parsed_stmt = ParseStmt(stmt.cast<py::object>());
        if (parsed_stmt) {
            body.push_back(parsed_stmt);
        }
    }

    // Create With with constructor
    ast::With *with_stmt = context_->Allocate<ast::With>();
    new (with_stmt) ast::With(std::move(items), std::move(body));
    return with_stmt;
}

ast::Pass *AveLangParser::ParsePass(const py::object &py_pass) {
    // Create Pass with constructor
    ast::Pass *pass_stmt = context_->Allocate<ast::Pass>();
    new (pass_stmt) ast::Pass();
    return pass_stmt;
}

ast::Break *AveLangParser::ParseBreak(const py::object &py_break) {
    // Create Break with constructor
    ast::Break *break_stmt = context_->Allocate<ast::Break>();
    new (break_stmt) ast::Break();
    return break_stmt;
}

ast::Continue *AveLangParser::ParseContinue(const py::object &py_continue) {
    // Create Continue with constructor
    ast::Continue *continue_stmt = context_->Allocate<ast::Continue>();
    new (continue_stmt) ast::Continue();
    return continue_stmt;
}

ast::ExprStmt *AveLangParser::ParseExprStmt(const py::object &py_expr) {
    // Parse the value (Python Expr nodes have a 'value' attribute for
    // expression statements)
    ast::Expr *value = nullptr;
    if (py::hasattr(py_expr, "value")) {
        py::object value_obj = py_expr.attr("value");
        value = ParseExpr(value_obj);
    }

    // Create ExprStmt with constructor
    ast::ExprStmt *expr_stmt = context_->Allocate<ast::ExprStmt>();
    new (expr_stmt) ast::ExprStmt(value);
    return expr_stmt;
}

ast::Import *AveLangParser::ParseImport(const py::object &py_import) {
    // Parse names (list of alias objects)
    llvm::SmallVector<ast::Alias, 4> names;
    for (auto name_obj : MaybePyList(py_import, "names")) {
        py::object alias_obj = name_obj.cast<py::object>();

        // Parse name
        std::string name;
        if (py::hasattr(alias_obj, "name")) {
            name = alias_obj.attr("name").cast<std::string>();
        }

        // Parse asname (optional)
        std::string asname;
        if (py::hasattr(alias_obj, "asname") &&
            !alias_obj.attr("asname").is_none()) {
            asname = alias_obj.attr("asname").cast<std::string>();
        }

        names.emplace_back(std::move(name), std::move(asname));
    }

    // Create Import with constructor
    ast::Import *import_stmt = context_->Allocate<ast::Import>();
    new (import_stmt) ast::Import(std::move(names));
    return import_stmt;
}

ast::ImportFrom *
AveLangParser::ParseImportFrom(const py::object &py_import_from) {
    // Parse module name
    std::string module;
    if (py::hasattr(py_import_from, "module") &&
        !py_import_from.attr("module").is_none()) {
        module = py_import_from.attr("module").cast<std::string>();
    }

    // Parse names (list of alias objects)
    llvm::SmallVector<ast::Alias, 4> names;
    for (auto name_obj : MaybePyList(py_import_from, "names")) {
        py::object alias_obj = name_obj.cast<py::object>();

        // Parse name
        std::string name;
        if (py::hasattr(alias_obj, "name")) {
            name = alias_obj.attr("name").cast<std::string>();
        }

        // Parse asname (optional)
        std::string asname;
        if (py::hasattr(alias_obj, "asname") &&
            !alias_obj.attr("asname").is_none()) {
            asname = alias_obj.attr("asname").cast<std::string>();
        }

        names.emplace_back(std::move(name), std::move(asname));
    }

    // Parse level (for relative imports)
    int level = 0;
    if (py::hasattr(py_import_from, "level")) {
        level = py_import_from.attr("level").cast<int>();
    }

    // Create ImportFrom with constructor
    ast::ImportFrom *import_from_stmt = context_->Allocate<ast::ImportFrom>();
    new (import_from_stmt)
        ast::ImportFrom(std::move(module), std::move(names), level);
    return import_from_stmt;
}

} // namespace causalflow::avelang::frontend
