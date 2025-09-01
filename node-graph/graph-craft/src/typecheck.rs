use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use crate::document::{InlineRust, value};
use crate::document::{NodeId, OriginalLocation};
use crate::proto::{ConstructionArgs, GraphError, GraphErrorType, GraphErrors, ProtoNetwork, ProtoNode};

pub use graphene_core::registry::*;
use graphene_core::*;

////////////////////////////////////////////////////////////////////////////////

/// The `TypingContext` is used to store the types of the nodes indexed by their stable node id.
#[derive(Default, Clone, dyn_any::DynAny)]
pub struct TypingContext {
	lookup: Cow<'static, HashMap<ProtoNodeIdentifier, HashMap<NodeIOTypes, NodeConstructor>>>,
	inferred: HashMap<NodeId, NodeIOTypes>,
	constructor: HashMap<NodeId, NodeConstructor>,
}

impl TypingContext {
	/// Creates a new `TypingContext` with the given lookup table.
	pub fn new(lookup: &'static HashMap<ProtoNodeIdentifier, HashMap<NodeIOTypes, NodeConstructor>>) -> Self {
		Self {
			lookup: Cow::Borrowed(lookup),
			..Default::default()
		}
	}

	/// Updates the `TypingContext` with a given proto network. This will infer the types of the nodes
	/// and store them in the `inferred` field. The proto network has to be topologically sorted
	/// and contain fully resolved stable node ids.
	pub fn update(&mut self, network: &ProtoNetwork) -> Result<(), GraphErrors> {
		for (id, node) in network.nodes.iter() {
			self.infer(*id, node)?;
		}

		Ok(())
	}

	pub fn remove_inference(&mut self, node_id: NodeId) -> Option<NodeIOTypes> {
		self.constructor.remove(&node_id);
		self.inferred.remove(&node_id)
	}

	/// Returns the node constructor for a given node id.
	pub fn constructor(&self, node_id: NodeId) -> Option<NodeConstructor> {
		self.constructor.get(&node_id).copied()
	}

	/// Returns the type of a given node id if it exists
	pub fn type_of(&self, node_id: NodeId) -> Option<&NodeIOTypes> {
		self.inferred.get(&node_id)
	}

	/// Returns the inferred types for a given node id.
	pub fn infer(&mut self, node_id: NodeId, node: &ProtoNode) -> Result<NodeIOTypes, GraphErrors> {
		// Return the inferred type if it is already known
		if let Some(inferred) = self.inferred.get(&node_id) {
			return Ok(inferred.clone());
		}

		let inputs = match node.construction_args {
			// If the node has a value input we can infer the return type from it
			ConstructionArgs::Value(ref v) => {
				// TODO: This should return a reference to the value
				let types = NodeIOTypes::new(concrete!(Context), Type::Future(Box::new(v.ty())), vec![]);
				self.inferred.insert(node_id, types.clone());
				return Ok(types);
			}
			// If the node has nodes as inputs we can infer the types from the node outputs
			ConstructionArgs::Nodes(ref nodes) => nodes
				.iter()
				.map(|id| {
					self.inferred
						.get(id)
						.ok_or_else(|| vec![GraphError::new(node, GraphErrorType::NodeNotFound(*id))])
						.map(|node| node.ty())
				})
				.collect::<Result<Vec<Type>, GraphErrors>>()?,
			ConstructionArgs::Inline(ref inline) => vec![inline.ty.clone()],
		};

		// Get the node input type from the proto node declaration
		let call_argument = &node.call_argument;
		let impls = self.lookup.get(&node.identifier).ok_or_else(|| vec![GraphError::new(node, GraphErrorType::NoImplementations)])?;

		if let Some(index) = inputs.iter().position(|p| {
			matches!(p,
			Type::Fn(_, b) if matches!(b.as_ref(), Type::Generic(_)))
		}) {
			return Err(vec![GraphError::new(node, GraphErrorType::UnexpectedGenerics { index, inputs })]);
		}

		/// Checks if a proposed input to a particular (primary or secondary) input connector is valid for its type signature.
		/// `from` indicates the value given to a input, `to` indicates the input's allowed type as specified by its type signature.
		fn valid_type(from: &Type, to: &Type) -> bool {
			match (from, to) {
				// Direct comparison of two concrete types.
				(Type::Concrete(type1), Type::Concrete(type2)) => type1 == type2,
				// Check inner type for futures
				(Type::Future(type1), Type::Future(type2)) => valid_type(type1, type2),
				// Direct comparison of two function types.
				// Note: in the presence of subtyping, functions are considered on a "greater than or equal to" basis of its function type's generality.
				// That means we compare their types with a contravariant relationship, which means that a more general type signature may be substituted for a more specific type signature.
				// For example, we allow `T -> V` to be substituted with `T' -> V` or `() -> V` where T' and () are more specific than T.
				// This allows us to supply anything to a function that is satisfied with `()`.
				// In other words, we are implementing these two relations, where the >= operator means that the left side is more general than the right side:
				// - `T >= T' ⇒ (T' -> V) >= (T -> V)` (functions are contravariant in their input types)
				// - `V >= V' ⇒ (T -> V) >= (T -> V')` (functions are covariant in their output types)
				// While these two relations aren't a truth about the universe, they are a design decision that we are employing in our language design that is also common in other languages.
				// For example, Rust implements these same relations as it describes here: <https://doc.rust-lang.org/nomicon/subtyping.html>
				// Graphite doesn't have subtyping currently, but it used to have it, and may do so again, so we make sure to compare types in this way to make things easier.
				// More details explained here: <https://github.com/GraphiteEditor/Graphite/issues/1741>
				(Type::Fn(in1, out1), Type::Fn(in2, out2)) => valid_type(out2, out1) && valid_type(in1, in2),
				// If either the proposed input or the allowed input are generic, we allow the substitution (meaning this is a valid subtype).
				// TODO: Add proper generic counting which is not based on the name
				(Type::Generic(_), _) | (_, Type::Generic(_)) => true,
				// Reject unknown type relationships.
				_ => false,
			}
		}

		// List of all implementations that match the input types
		let valid_output_types = impls
			.keys()
			.filter(|node_io| valid_type(&node_io.call_argument, call_argument) && inputs.iter().zip(node_io.inputs.iter()).all(|(p1, p2)| valid_type(p1, p2)))
			.collect::<Vec<_>>();

		// Attempt to substitute generic types with concrete types and save the list of results
		let substitution_results = valid_output_types
			.iter()
			.map(|node_io| {
				let generics_lookup: Result<HashMap<_, _>, _> = collect_generics(node_io)
					.iter()
					.map(|generic| check_generic(node_io, call_argument, &inputs, generic).map(|x| (generic.to_string(), x)))
					.collect();

				generics_lookup.map(|generics_lookup| {
					let orig_node_io = (*node_io).clone();
					let mut new_node_io = orig_node_io.clone();
					replace_generics(&mut new_node_io, &generics_lookup);
					(new_node_io, orig_node_io)
				})
			})
			.collect::<Vec<_>>();

		// Collect all substitutions that are valid
		let valid_impls = substitution_results.iter().filter_map(|result| result.as_ref().ok()).collect::<Vec<_>>();

		match valid_impls.as_slice() {
			[] => {
				let mut best_errors = usize::MAX;
				let mut error_inputs = Vec::new();
				for node_io in impls.keys() {
					let current_errors = [call_argument]
						.into_iter()
						.chain(&inputs)
						.cloned()
						.zip([&node_io.call_argument].into_iter().chain(&node_io.inputs).cloned())
						.enumerate()
						.filter(|(_, (p1, p2))| !valid_type(p1, p2))
						.map(|(index, ty)| {
							let i = node.original_location.inputs(index).min_by_key(|s| s.node.len()).map(|s| s.index).unwrap_or(index);
							(i, ty)
						})
						.collect::<Vec<_>>();
					if current_errors.len() < best_errors {
						best_errors = current_errors.len();
						error_inputs.clear();
					}
					if current_errors.len() <= best_errors {
						error_inputs.push(current_errors);
					}
				}
				let inputs = [call_argument]
					.into_iter()
					.chain(&inputs)
					.enumerate()
					.filter_map(|(i, t)| if i == 0 { None } else { Some(format!("• Input {i}: {t}")) })
					.collect::<Vec<_>>()
					.join("\n");
				Err(vec![GraphError::new(node, GraphErrorType::InvalidImplementations { inputs, error_inputs })])
			}
			[(node_io, org_nio)] => {
				let node_io = node_io.clone();

				// Save the inferred type
				self.inferred.insert(node_id, node_io.clone());
				self.constructor.insert(node_id, impls[org_nio]);
				Ok(node_io)
			}
			// If two types are available and one of them accepts () an input, always choose that one
			[first, second] => {
				if first.0.call_argument != second.0.call_argument {
					for (node_io, orig_nio) in [first, second] {
						if node_io.call_argument != concrete!(()) {
							continue;
						}

						// Save the inferred type
						self.inferred.insert(node_id, node_io.clone());
						self.constructor.insert(node_id, impls[orig_nio]);
						return Ok(node_io.clone());
					}
				}
				let inputs = [call_argument].into_iter().chain(&inputs).map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
				let valid = valid_output_types.into_iter().cloned().collect();
				Err(vec![GraphError::new(node, GraphErrorType::MultipleImplementations { inputs, valid })])
			}

			_ => {
				let inputs = [call_argument].into_iter().chain(&inputs).map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
				let valid = valid_output_types.into_iter().cloned().collect();
				Err(vec![GraphError::new(node, GraphErrorType::MultipleImplementations { inputs, valid })])
			}
		}
	}
}

/// Returns a list of all generic types used in the node
fn collect_generics(types: &NodeIOTypes) -> Vec<Cow<'static, str>> {
	let inputs = [&types.call_argument].into_iter().chain(types.inputs.iter().map(|x| x.nested_type()));
	let mut generics = inputs
		.filter_map(|t| match t {
			Type::Generic(out) => Some(out.clone()),
			_ => None,
		})
		.collect::<Vec<_>>();
	if let Type::Generic(out) = &types.return_value {
		generics.push(out.clone());
	}
	generics.dedup();
	generics
}

/// Checks if a generic type can be substituted with a concrete type and returns the concrete type
fn check_generic(types: &NodeIOTypes, input: &Type, parameters: &[Type], generic: &str) -> Result<Type, String> {
	let inputs = [(Some(&types.call_argument), Some(input))]
		.into_iter()
		.chain(types.inputs.iter().map(|x| x.fn_input()).zip(parameters.iter().map(|x| x.fn_input())))
		.chain(types.inputs.iter().map(|x| x.fn_output()).zip(parameters.iter().map(|x| x.fn_output())));
	let concrete_inputs = inputs.filter(|(ni, _)| matches!(ni, Some(Type::Generic(input)) if generic == input));
	let mut outputs = concrete_inputs.flat_map(|(_, out)| out);
	let out_ty = outputs
		.next()
		.ok_or_else(|| format!("Generic output type {generic} is not dependent on input {input:?} or parameters {parameters:?}",))?;
	if outputs.any(|ty| ty != out_ty) {
		return Err(format!("Generic output type {generic} is dependent on multiple inputs or parameters",));
	}
	Ok(out_ty.clone())
}

/// Returns a list of all generic types used in the node
fn replace_generics(types: &mut NodeIOTypes, lookup: &HashMap<String, Type>) {
	let replace = |ty: &Type| {
		let Type::Generic(ident) = ty else {
			return None;
		};
		lookup.get(ident.as_ref()).cloned()
	};
	types.call_argument.replace_nested(replace);
	types.return_value.replace_nested(replace);
	for input in &mut types.inputs {
		input.replace_nested(replace);
	}
}
